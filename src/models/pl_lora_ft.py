import lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from pytorch_optimizer import Ranger21
from src.helpers.scores import select_choices
from einops import rearrange
from transformers.modeling_outputs import ModelOutput
from src.helpers.torch import clear_mem, detachcpu, recursive_copy

def hacky_sanitize_outputs(o):
    """I can't find the mem leak, so lets just detach, cpu, clone, freemem."""
    o = {k: detachcpu(v) for k, v in o.items()}
    o = recursive_copy(o, detach=True, clone=True)
    clear_mem()
    return o

def postprocess_result(i, o):

    # note that the results are huge. It might be worth convertting to int16 or similar so we can save to disc as we go https://github.com/EleutherAI/elk/blob/84e99a36a5050881d85f1510a2486ce46ac1f942/elk/utils/typing.py#L16
    assert torch.isfinite(o['logits']).all()
    # hidden states come at as lists of layers, lets stack them
    hidden_states = rearrange(list(o['hidden_states']), 'l b t h -> b l t h').detach().cpu().float()
    end_hidden_states = hidden_states[:, :, -1, :]
    end_residual_stream = end_hidden_states.diff(1)

    end_logits = o["logits"][:, -1].detach().cpu().float()
    probs = torch.softmax(end_logits, -1)
    choice_ids = i['choice_ids'].detach().cpu().long()


    choice_probs = select_choices(probs, choice_ids).sum(2)

    # shape[choices, intervention_version]
    binary_ans = choice_probs[:, 1] / (choice_probs.sum(1) + 1e-12)

    o = dict(
        end_residual_stream=end_residual_stream, 
        end_logits=end_logits,

        # maybe these ones should be postprocessing
        choice_probs=choice_probs,
        binary_ans=binary_ans,
        label_true=i['label_true'],
        instructed_to_lie=i['instructed_to_lie'],
        sys_instr_name=i['sys_instr_name'],
        example_i=i['example_i'],
    )

    # why oh why do I get mem leaks like this
    o = hacky_sanitize_outputs(o)

    # we only want the last token
    o = ModelOutput(
        **o
    )
    return o


def get_loss(batch, out, out_a):
    """
    loss which encourages it to switch it's answers with the base model
    """

    log_probs_a = torch.log_softmax(out_a["logits"][:, -1,], -1,)
    log_probs = torch.log_softmax(out["logits"][:, -1,], -1,)

    # switched probs for our choices (e.g. Yes <> No)
    id_neg = batch["choice_ids"][:, 0].clone()
    id_pos = batch["choice_ids"][:, 1].clone()

    log_probs_r = log_probs.clone()
    for i in range(id_neg.shape[1]):
        log_probs_r[:, id_neg[:, i]] = log_probs[:, id_pos[:, i]]
        log_probs_r[:, id_pos[:, i]] = log_probs[:, id_neg[:, i]]
    log_probs_r = log_probs_r.detach()

    # Either just optimise for choice probs...
    choice_lprobs_a = select_choices(log_probs_a, batch["choice_ids"].clone())#.sum(2)
    choice_lprobs_r = select_choices(log_probs_r, batch["choice_ids"].clone())#.sum(2)
    choice_lprobs_r = choice_lprobs_r.detach()
    loss_choices = F.kl_div(
        choice_lprobs_a, choice_lprobs_r, reduction="batchmean", log_target=True
    )

    # or constrain on all probs or just choices?
    loss_all = F.kl_div(
        log_probs_a, log_probs_r, reduction="batchmean", log_target=True
    )
    loss = loss_choices # + loss_all * 1e-8
    # loss = loss_all

    assert torch.isfinite(loss)

    return loss, loss_choices, loss_all


class AtapterFinetuner(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        total_steps: int = 1,
        lr=4e-3,
        weight_decay=1e-9,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(
            ignore=["model", "tokenizer"],
        )

    def forward(self, b):
        b_in = dict(
            input_ids=b["input_ids"].clone(),
            attention_mask=b["attention_mask"].clone(),
        )

        # handled by accelerator
        # b_in = {k: v.to(self.model.device) for k, v in b_in.items()}

        o = self.model(
            **b_in, use_cache=False, output_hidden_states=True, return_dict=True
        )
        return o

    def _step(self, batch, batch_idx=0, stage="train"):
        with torch.no_grad():
            with self.model.disable_adapter():
                out = self(batch)

        # self.model.enable_adapters()
        out_a = self(batch)

        if stage == "pred":
            res = {f'{k}_base':v for k,v in postprocess_result(batch, out).items()}
            res_a = {f'{k}_adapt':v for k,v in postprocess_result(batch, out_a).items()}
            res = dict(**res, **res_a)
            res_a = out = out_a = None
            clear_mem()
            return res
        
        loss, loss_choices, loss_all = get_loss(batch, out, out_a)
        assert torch.isfinite(loss)

        batch_size = batch["input_ids"].shape[0]
        self.log(f"{stage}/loss",loss, on_epoch=True, on_step=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage="val")

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        with torch.no_grad():
            return self._step(batch, batch_idx, stage="pred")

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """use ranger21 from  https://github.com/kozistr/pytorch_optimizer"""
        assert self.hparams.total_steps>1
        optimizer = Ranger21(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            num_iterations=self.hparams.total_steps,
        )
        return optimizer

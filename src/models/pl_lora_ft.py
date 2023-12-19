import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from pytorch_optimizer import Ranger21
from src.helpers.scores import get_choice_probs, logits2choice_probs2
from einops import rearrange
from transformers.modeling_outputs import ModelOutput


def postprocess_result(i, o):
    assert torch.isfinite(o['logits']).all()
    # hidden states come at as lists of layers, lets stack them
    hidden_states = rearrange(list(o['hidden_states']), 'l b t h -> b l t h')
    end_hidden_states = hidden_states[:, -1]
    end_logits = o["logits"][:, -1]


    # choice probs
    ii = end_logits.shape[1]
    choice_probs = torch.stack(
        [
            logits2choice_probs2(end_logits[:, j], i["choice_ids"])
            for j in range(ii)
        ],
        1,
    )

    # shape[choices, intervention_version]
    binary_ans = choice_probs[1] / (torch.sum(choice_probs, 0) + 1e-12)

    # we only want the last token
    o = ModelOutput(
        end_hidden_states=end_hidden_states, 
        end_logits=end_logits,

        # maybe these ones should be postprocessing
        choice_probs=choice_probs,
        binary_ans=binary_ans,
        label_true=i['label_true'],
        instructed_to_lie=i['instructed_to_lie'],
    )
    return o


def get_loss(batch, out, out_a):

    log_probs_a = torch.log_softmax(out_a["logits"][:,-1,],-1,)
    log_probs = torch.log_softmax(out["logits"][:,-1,],-1,)

    # get loss, so that our adapter returns switched probs for our choices (e.g. Yes <> No)
    id_neg = batch["choice_ids"][:, 0]
    id_pos = batch["choice_ids"][:, 1]

    log_probs_r = log_probs.clone()
    for i in range(id_neg.shape[1]):
        log_probs_r[:, id_neg[:, i]] = log_probs[:, id_pos[:, i]]
        log_probs_r[:, id_pos[:, i]] = log_probs[:, id_neg[:, i]]

    # Either just optimise for choice probs...
    choice_probs_a = get_choice_probs(log_probs_a, batch["choice_ids"])
    choice_probs_r = get_choice_probs(log_probs_r, batch["choice_ids"])
    loss_choices = F.kl_div(
        choice_probs_a, choice_probs_r, reduction="batchmean", log_target=True
    )

    # or constrain on all probs or just choices?
    loss_all = F.kl_div(
        log_probs_a, log_probs_r, reduction="batchmean", log_target=True
    )

    loss = loss_choices  # + loss_all / 100
    return loss, loss_choices, loss_all


class AtapterFinetuner(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        total_steps: int,
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
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"],
        )
        b_in = {k: v.to(self.model.device) for k, v in b_in.items()}

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
            res = {f'{k}_base':v for k,v in postprocess_result(batch, out)}
            res_a = {f'{k}_adapt':v for k,v in postprocess_result(batch, out_a)}
            return dict(
                **res, **res_a
            )
        
        loss, loss_choices, loss_all = get_loss(batch, out, out_a)

        batch_size = batch["input_ids"].shape[0]
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/loss_choices": loss_choices,
                f"{stage}/loss_all": loss_all,
            },
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage="val")

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage="pred").cpu().detach()

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self._step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """use ranger21 from  https://github.com/kozistr/pytorch_optimizer"""
        optimizer = Ranger21(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            num_iterations=self.hparams.total_steps,
        )
        return optimizer

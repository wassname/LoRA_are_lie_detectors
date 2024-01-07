import lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from pytorch_optimizer import Ranger21
from src.helpers.scores import select_choices
from einops import rearrange
from transformers.modeling_outputs import ModelOutput
from jaxtyping import Float, Int
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union
from baukit.nethook import TraceDict
from torch import optim
from src.helpers.torch_helpers import clear_mem, detachcpu, recursive_copy, switch
import bitsandbytes as bnb

def hacky_sanitize_outputs(o):
    """I can't find the mem leak, so lets just detach, cpu, clone, free mem."""
    o = {k: detachcpu(v) for k, v in o.items()}
    o = recursive_copy(o, detach=True, clone=True)
    clear_mem()
    return o


def postprocess_result(input: dict, ret: TraceDict, output: ModelOutput, get_residual=True) -> ModelOutput:

    # note that the results are huge. It might be worth convertting to int16 or similar so we can save to disc as we go https://github.com/EleutherAI/elk/blob/84e99a36a5050881d85f1510a2486ce46ac1f942/elk/utils/typing.py#L16
    assert torch.isfinite(output['logits']).all()
    

    end_logits = output["logits"][:, -1].detach().cpu().float()
    probs = torch.softmax(end_logits, -1)
    choice_ids = input['choice_ids'].detach().cpu().long()

    label_instructed = input['label_true'] ^ input['instructed_to_lie']


    choice_probs = select_choices(probs, choice_ids).sum(2)

    # shape[choices, intervention_version]
    binary_ans = choice_probs[:, 1] / (choice_probs.sum(1) + 1e-12)


    correct_truth_telling = switch(binary_ans, input['label_true'])
    correct_instruction_following = switch(binary_ans, label_instructed)

    out = dict(
        end_logits=end_logits,

        # maybe these ones should be postprocessing
        choice_probs=choice_probs,
        binary_ans=binary_ans,
        label_true=input['label_true'],
        label_instructed=label_instructed,
        instructed_to_lie=input['instructed_to_lie'],
        sys_instr_name=input['sys_instr_name'],
        example_i=input['example_i'],
        ds_string=input['ds_string'],
        template_name=input['template_name'],
        correct_truth_telling=correct_truth_telling,
        correct_instruction_following=correct_instruction_following,
    )
    if get_residual:
        # we can also get activations from layers monitored in baukit
        activations = {}
        for k in ret.keys():
            suffix = k.split('.')[-1]
            if suffix not in activations:
                activations[suffix] = []
            activations[suffix].append(ret[k].output)

        for k in activations.keys():
            # HACK: we will assume they are all shaped [batch, tokens, hidden]
            activation = rearrange(activations[k], 'l b t h -> b l t h').detach().cpu().float()
            end_activation = activation[:, :, -1, :]
            # # Diff normally removes the first layer. Let's keep it... but only when using something that has the residual added in
            # end_residual = end_activation.diff(dim=1, prepend=torch.zeros_like(end_activation)[:, :1])
            out[f'end_residual_{k}'] = end_activation

        # ret = {k: v.detach().cpu().float() for k, v in ret.items()}
        

        # # hidden states come at as lists of layers, lets stack them
        # hidden_states = rearrange(list(output['hidden_states']), 'l b t h -> b l t h').detach().cpu().float()
        # end_hidden_states = hidden_states[:, :, -1, :]
        # end_residual_stream = end_hidden_states.diff(1)
        # out['end_residual_stream'] = end_residual_stream

    # why oh why do I get mem leaks like this
    out = hacky_sanitize_outputs(out)

    # we only want the last token
    out = ModelOutput(
        **out
    )
    return out





class AtapterFinetuner(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        total_steps: int = 1,
        lr=4e-3,
        weight_decay=1e-9,
        collection_layers:list=[]
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
    
    def get_loss(self, batch, out, out_a):
        """
        loss which encourages it to switch it's answers with the base model
        """

        log_probs_a = torch.log_softmax(out_a["logits"][:, -1,], -1,)
        log_probs = torch.log_softmax(out["logits"][:, -1,], -1,)

        # switched probs for our choices (e.g. Yes <> No)
        id_neg = batch["choice_ids"][:, 0].clone()
        id_pos = batch["choice_ids"][:, 1].clone()

        log_probs_r = log_probs.clone()

        # batch['instructed_to_lie']
        # TODO try making it a perfect lier, e.g. when instructed to lie
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

    def _step(self, batch, batch_idx=0, stage="train"):

        if stage == "pred":
            # FIXME, not used in collect
            with TraceDict(self.model, self.hparams.collection_layers, detach=True) as ret:
                with self.model.disable_adapter():
                    out = self(batch)
            with TraceDict(self.model, self.hparams.collection_layers, detach=True) as ret_a:
                out_a = self(batch)
            res = {f'{k}_base':v for k,v in postprocess_result(batch, ret, out).items()}
            res_a = {f'{k}_adapt':v for k,v in postprocess_result(batch, ret_a, out_a).items()}
            res = dict(**res, **res_a)
            res_a = out = out_a = None
            clear_mem()
            return res
        
        with torch.no_grad():
            with self.model.disable_adapter():
                out = self(batch)

        # self.model.enable_adapters()
        out_a = self(batch)
        
        loss, loss_choices, loss_all = self.get_loss(batch, out, out_a)
        assert torch.isfinite(loss)

        batch_size = batch["input_ids"].shape[0]
        self.log(f"{stage}/loss",loss, on_epoch=True, on_step=True, batch_size=batch_size, 
        prog_bar=True)
        self.log(f"{stage}/n", batch_size* 1.0, on_epoch=True, on_step=False, reduce_fx=torch.sum)
        # loss diff with old?
        self.log(f"{stage}/logits_diff", torch.abs(out.logits - out.logits).mean(), on_step=True, batch_size=batch_size)
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

    # def configure_optimizers(self):
    #     """use ranger21 from  https://github.com/kozistr/pytorch_optimizer"""
    #     assert self.hparams.total_steps>1
    #     optimizer = Ranger21(
    #         self.parameters(),
    #         lr=self.hparams.lr,
    #         weight_decay=self.hparams.weight_decay,
    #         num_iterations=self.hparams.total_steps,
    #     )
    #     return optimizer
    
    def configure_optimizers(self):
        """simple vanilla torch optim"""
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html#quantization-via-bitsandbytes
        # optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, self.hparams.lr, total_steps=self.hparams.total_steps, verbose=False,
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import torch
from datasets import Dataset
import datasets
from tqdm.auto import tqdm
from torch.utils.data import random_split, DataLoader, TensorDataset
import json
from loguru import logger
from src.helpers.torch import clear_mem, detachcpu, recursive_copy
from src.models.pl_lora_ft import postprocess_result
from src.config import root_folder

@torch.no_grad
def generate_batches(loader: DataLoader, model: AutoModelForCausalLM, get_residual=True) -> dict:

    model.eval()
    for batch in tqdm(loader, 'collecting hidden states'):
        b_in = dict(
            input_ids=batch["input_ids"].clone(),
            attention_mask=batch["attention_mask"].clone(),
        )
        if hasattr(model, 'disable_adapter'):
            with model.disable_adapter():
                out = model(**b_in, use_cache=False, output_hidden_states=True, return_dict=True)
                res = {f'{k}_base':v for k,v in postprocess_result(batch, out, get_residual=get_residual).items()}
                del out
            out_a = model(**b_in, use_cache=False, output_hidden_states=True, return_dict=True)
            res_a = {f'{k}_adapt':v for k,v in postprocess_result(batch, out_a, get_residual=get_residual).items()}
            del out_a
        else:
            out = model(**b_in, use_cache=False, output_hidden_states=True, return_dict=True)
            res = {f'{k}_base':v for k,v in postprocess_result(batch, out, get_residual=get_residual).items()}
            res_a = {}
            logger.warning("model does not have disable_adapter")


        o = dict(**res, **res_a)
        res = res_a = out = out_a = b_in = None
        o = recursive_copy(o)
        clear_mem()
        yield o


from datasets.arrow_writer import ArrowWriter 
from datasets.fingerprint import Hasher

def ds_hash(**kwargs):
    suffix = Hasher.hash(kwargs)
    return suffix


def manual_collect2(loader: DataLoader, model: AutoModelForCausalLM, dataset_name='', get_residual=True):
    hash = ds_hash(generate_batches=generate_batches, loader=loader, model=model)
    f = root_folder / ".ds" / f"ds_{dataset_name}_{hash}"
    f.parent.mkdir(exist_ok=True, parents=True)
    f = str(f)
    logger.info(f"creating dataset {f}")
    iterator = generate_batches(loader, model, get_residual=get_residual)
    with ArrowWriter(path=f, writer_batch_size=6) as writer: 
        for bo in iterator:
            # dict_of_batches_to_batch_of_dicts 
            # {k: (v.shape, v.dtype, v.device) for k,v in o.items()}
            boT = [{k: bo[k][i] for k in bo.keys()} for i in range(len(bo['label_true_base']))]
            for o in boT:
                writer.write(o)
        writer.finalize() 
    
    ds = Dataset.from_file(f).with_format("torch")
    return ds, f

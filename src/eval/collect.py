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
def generate_batches(loader: DataLoader, model: AutoModelForCausalLM) -> dict:

    model.eval()
    for batch in tqdm(loader, 'collecting hidden states'):
        b_in = dict(
            input_ids=batch["input_ids"].clone(),
            attention_mask=batch["attention_mask"].clone().half(),
        )
        with model.disable_adapter():
            out = model(**b_in, use_cache=False, output_hidden_states=True, return_dict=True)
            res = {f'{k}_base':v for k,v in postprocess_result(batch, out).items()}
            del out

        out_a = model(**b_in, use_cache=False, output_hidden_states=True, return_dict=True)
        res_a = {f'{k}_adapt':v for k,v in postprocess_result(batch, out_a).items()}
        del out_a

        o = dict(**res, **res_a)
        res = res_a = out = out_a = b_in = None
        o = recursive_copy(o)
        clear_mem()
        yield o


from datasets.arrow_writer import ArrowWriter 
from datasets.fingerprint import Hasher

def ds_hash(generate_batches, loader, model):
    suffix = Hasher.hash(dict(
        generate_batches=generate_batches,
        model=model,
        loader=loader,
    ))
    return suffix


def manual_collect2(loader: DataLoader, model: AutoModelForCausalLM, dataset_name='', split_type="train", info_kwargs={}):
    hash = ds_hash(generate_batches, loader, model)
    f = root_folder / ".ds" / f"ds_{dataset_name}_{hash}"
    f.parent.mkdir(exist_ok=True, parents=True)
    f = str(f)
    logger.info(f"creating dataset {f}")
    iterator = generate_batches(loader, model)
    with ArrowWriter(path=f, writer_batch_size=6) as writer: 
        for bo in iterator:
            # dict_of_batches_to_batch_of_dicts 
            # {k: (v.shape, v.dtype, v.device) for k,v in o.items()}
            boT = [{k: bo[k][i] for k in bo.keys()} for i in range(len(bo['label_true_base']))]
            for o in boT:
                writer.write(o)
        writer.finalize() 
    
    ds = Dataset.from_file(f)   
    return ds, f


# def manual_collect(loader: DataLoader, model: AutoModelForCausalLM, dataset_name='', split_type="train", info_kwargs={}):
#     """the hidden states are huge so we have to cache to disk. we can do this using datasets.Dataset.from_generator"""
#     # root_folder = Path("~/.cache/huggingface/datasets")
#     f = str(root_folder / ".ds" / f"{dataset_name}")
#     # generator = generate_batches(loader, model)
#     gen_kwargs = dict(loader=loader, model=model)
#     ds_out = Dataset.from_generator(
#         generator=generate_batches,
#         info=datasets.DatasetInfo(
#             description=json.dumps(info_kwargs, indent=2),
#             config_name=f,
#         ),
#         gen_kwargs=gen_kwargs,
#         # features=dataset_features,
#         num_proc=1,
#         # split=split_type,
#     )
#     logger.info(f"Created dataset {dataset_name} with {len(loader.dataset)} examples at `{f}`")
#     ds_out.to_disk(f)
#     # ds_out = Dataset.from_dict({k: torch.concat([rr[k] for rr in data]) for k in data[0].keys()}).with_format("torch")
#     return ds_out, f

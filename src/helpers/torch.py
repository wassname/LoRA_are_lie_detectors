import torch
import numpy as np
import transformers
import random
import gc
import pandas as pd

def get_top_n(scores: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer, n=10) -> pd.Series:
    """Get top n choices and their probabilities given raw logits"""
    probs = scores.softmax(-1).squeeze()
    assert len(probs.shape)==1
    top10 = torch.argsort(probs, dim=-1, descending=True)[:n]
    top10_probs = probs[top10]
    top10_ext = tokenizer.batch_decode(top10)
    return pd.Series(top10_probs, index=top10_ext, name='probs')

def to_numpy(x):
    """
    Trys to convert torch to numpy and if possible a single item
    """
    if isinstance(x, torch.Tensor):
        # note apache parquet doesn't support half https://github.com/huggingface/datasets/issues/4981
        x = x.detach().cpu().float()
        if x.squeeze().dim()==0:
            return x.item()
        return x.numpy()
    else:
        return x



def set_seeds(n: int) -> None:
    transformers.set_seed(n)
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)
    
def to_item(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    return x

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def detachcpu(x):
    """
    Trys to convert torch if possible a single item
    """
    if isinstance(x, torch.Tensor):
        # note apache parquet doesn't support half to we go for float https://github.com/huggingface/datasets/issues/4981
        x = x.detach().cpu()
        if x.squeeze().dim()==0:
            return x.item()
        return x
    else:
        return x

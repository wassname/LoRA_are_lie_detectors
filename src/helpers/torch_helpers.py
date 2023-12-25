import torch
# import numpy as np
# import transformers
# import random
import gc
# import pandas as pd

# def get_top_n(scores: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer, n=10) -> pd.Series:
#     """Get top n choices and their probabilities given raw logits"""
#     probs = scores.softmax(-1).squeeze()
#     assert len(probs.shape)==1
#     top10 = torch.argsort(probs, dim=-1, descending=True)[:n]
#     top10_probs = probs[top10]
#     top10_ext = tokenizer.batch_decode(top10)
#     return pd.Series(top10_probs, index=top10_ext, name='probs')

# def to_numpy(x):
#     """
#     Trys to convert torch to numpy and if possible a single item
#     """
#     if isinstance(x, torch.Tensor):
#         # note apache parquet doesn't support half https://github.com/huggingface/datasets/issues/4981
#         x = x.detach().cpu().float()
#         if x.squeeze().dim()==0:
#             return x.item()
#         return x.numpy()
#     else:
#         return x



# def set_seeds(n: int) -> None:
#     transformers.set_seed(n)
#     torch.manual_seed(n)
#     np.random.seed(n)
#     random.seed(n)
    
# def to_item(x):
#     if isinstance(x, torch.Tensor):
#         x = x.detach().cpu().item()
#     return x

def clear_mem():
    gc.collect()
    # get_accelerator().empty_cache()
    # accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()

def detachcpu(x):
    """
    Trys to convert torch if possible a single item
    """
    if isinstance(x, torch.Tensor):
        # note apache parquet doesn't support half to we go for float https://github.com/huggingface/datasets/issues/4981
        x = x.detach().cpu()
        # if x.squeeze().dim()==0:
        #     return x.item()
        return x
    else:
        return x

def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    from baukit
    
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."

def batch_to_device(b, device=None):
    """Move a batch to the device"""
    if isinstance(b, torch.Tensor):
        return b.to(device)
    elif isinstance(b, dict):
        return {k:batch_to_device(v, device=device) for k,v in b.items()}
    elif isinstance(b, (list, tuple)):
        return type(b)([batch_to_device(v, device=device) for v in b])
    else:
        return b
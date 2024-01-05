import torch
import gc
import copy
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor

def switch(p: Float[Tensor, ""], s: Float[Tensor, ""]):
    """if the true label is 0, we will flip our binary prediction around. so 25% becomes 75%. It's the rating of how correct our answer was from 0 to 1"""
    s = s.float()
    return (1 - s) * (1-p) + s * p


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
        x = x.cpu()
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
        return type(x)({k: recursive_copy(v, clone=clone, detach=detach, retain_grad=retain_grad) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, clone=clone, detach=detach, retain_grad=retain_grad) for v in x])
    else:
        return copy.deepcopy(x)

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

def shape_of_anything(v):
    if isinstance(v, (Tensor, np.ndarray)):
        return v.shape
    elif isinstance(v, dict):
        return {k:shape_of_anything(v) for k,v in v.items()}
    elif isinstance(v, list):
        return len(v)
    else:
        return 1


from typing import Optional, List, Tuple, Dict
import numpy as np
# import torch
# import torch.nn.functional as F
import functools
import itertools
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel
)
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange

from src.helpers.select import select, select2

# default_class2choices = [['No', 'Negative', 'negative', 'no', 'false', 'wrong', 'False', '0'], ['Yes', 'Positive', 'positive', 'yes', 'true', 'correct', 'right', 'True', '1']]

def select_choices(end_logits: Float[Tensor, "batch tokens"], choices: Int[Tensor, "batch choices alternates"],) -> Float[Tensor, "batch choices * alternates"]:
    # batch_size = end_logits.shape[0]
    choices_flat = rearrange(choices, 'b c n -> b (c n)')

    # TODO unit test and split out next two lines
    # batch_range = torch.arange(batch_size).unsqueeze(0).to(choices_flat.device)
    # selected_logits = end_logits[batch_range, choices_flat]

    selected_logits = select2(end_logits, choices_flat)

    selected_logits = rearrange(selected_logits, 'b (c n) -> b c n', c=choices.shape[1])
    return selected_logits


@functools.lru_cache()
def choice2id(tokenizer, c: str, whitespace_first=False) -> List[int]:
    """convert a choice to a single token"""
    # HACK: this whole function is messy, and specific to the llama tokenizer :(. I don't want it to fail silently, so I'm adding a few asserts. It's better to find out before 4 hours of data collection
    
    # Note some tokenizers differentiate between "yes", "\nyes" and " yes", and ideally we want all! 
    ids2 = []
    ids2 += tokenizer(f' {c}', add_special_tokens=False)["input_ids"]
    ids2 += tokenizer(f'\n{c}', add_special_tokens=False)["input_ids"]
    ids2 += tokenizer(f'{c}', add_special_tokens=False)["input_ids"]
    ids = list(set(ids2))
    
    # only include ones that decode to our original
    ids = [i for i in ids if c.strip().startswith(tokenizer.decode(i).strip()) and len(tokenizer.decode(i).strip())]
    assert len(ids)
    
    # QC: they should all decode to the same token
    decoded_ids = tokenizer.batch_decode(ids)
    shortest = sorted(decoded_ids, key=lambda s:len(s))[0]
    assert len(shortest)
    assert all([decoded_ids[i].strip().startswith(shortest) for i in range(len(decoded_ids))]), f"decoded_ids={decoded_ids}"
    
    # check that we can decode it
    c3 = tokenizer.batch_decode(ids)
    for c2 in c3:
        if not c.strip().startswith(c2.strip()) and len(c2):
            print(c, c2, c3)
            ids = tokenizer(c, add_special_tokens=False)["input_ids"]
            decoded_ids = [tokenizer.decode(i).strip() for i in ids]
            print(f"{c}=>{ids}=>{decoded_ids}")
            raise AssertionError(f'We should be able to encode and decode the choices, but it failed: tokenizer.decode(tokenizer(`{c}`))==`{c2}`!=`{c}`')
    return ids

def choice2ids(all_choices: List[List[str]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    choices = [list(itertools.chain(*[choice2id(tokenizer, c) for c in choices])) for choices in all_choices]
    assert choices[0]!=choices[1], f"choices should be different but were not {all_choices}"
    assert choices[0][0]!=choices[1][0], "choices should be different"
    return choices


def row_choice_ids(r, tokenizer):
    return choice2ids([c for c in r['answer_choices']], tokenizer)


from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
import functools
import itertools
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel
)

default_class2choices = [['No', 'Negative', 'negative', 'no', 'false', 'wrong', 'False', '0'], ['Yes', 'Positive', 'positive', 'yes', 'true', 'correct', 'right', 'True', '1']]


def scores2choice_probs(row, class2_ids: List[List[int]], keys=["scores0"], prefix=""):
    """ Given next_token scores (logits) we take only the subset the corresponds to our
    - negative tokens (e.g. False, no, ...) 
    - and positive tokens (e.g. Yes, yes, affirmative, ...).
    
    example output:
    {'choice_probs1': array([0.39, 0.31 ], dtype=float32),
    'ans1': 0.44,
    'choice_probs2': array([0.44, 0.45], dtype=float32),
    'ans2': 0.502,}
    """
    eps = 1e-5
    out = {}
    for key in keys:
        scores = torch.tensor(row[key])
        probs = F.softmax(scores, 0).numpy() # shape [tokens, inferences)
        probs_c = [np.sum([probs[cc] for cc in c], 0) for c in class2_ids] # sum over alternate choices e.g. [['decrease', 'dec'],['inc', 'increase']]
        probs_c = np.stack(probs_c, 0) # shape [choices, inferences]
        
        # balance of probs
        out[prefix+key.replace("scores", "choice_probs")] = probs_c
        out[prefix+key.replace("scores", "ans")] = probs_c[1] / (np.sum(probs_c, 0) + eps) # shape is [inferences]

        # # balance of logits (much more exaggerated)
        # scores_c = [scores[class2_ids[c]].sum() for c in class2_ids]
        # out[key.replace("scores", "ansb")] = torch.tensor(scores_c).softmax(-1)[1].item()
    return out

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
    
    # print(ids2)
    # print(ids)
    # print([f'`{t}`' for t in tokenizer.batch_decode(ids, skip_special_tokens=True)])
    # print([c.strip().startswith(tokenizer.decode(i)) for i in ids])

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


def logits2choice_probs2(logits, choiceids: List[List[int]]):
    """calculate the probability for each group of choices."""
    assert logits.ndim==1, f"expected logits to be 1d, got {logits.shape}"
    assert logits.sum(0).abs()>2, 'pass in logits, not probs. you may have accidentally passed in a softmaxed values'
    choiceids = [list(set(i)) for i in choiceids] # we don't want to double count
    probs = torch.softmax(logits, 0)  # shape [tokens, inferences)
    probs_c = torch.tensor([[probs[cc] for cc in c] for c in choiceids]).sum(1)  # sum over alternate choices e.g. [['decrease', 'dec'],['inc', 'increase']]
    assert probs_c.sum()<=1.01, f"expected probs to sum to 1, got {probs_c.sum()}"
    return probs_c

def row_choice_ids(r, tokenizer):
    return choice2ids([c for c in r['answer_choices']], tokenizer)

from datasets import Dataset
from src.helpers.torch_helpers import clear_mem, detachcpu, recursive_copy, switch
from torch import Tensor
from jaxtyping import Float


def ds2label_model_obey(ds):
    """label: whether the model obeyed the instruction to lie"""
    label_instructed = ds["label_true_base"] ^ ds["instructed_to_lie_base"]
    ans = ds["binary_ans_base"] > 0.5
    labels_untruth = label_instructed == ans
    return labels_untruth


def ds2label_model_truth(ds):
    """label: whether the model told the truth"""
    ans = ds["binary_ans_base"] > 0.5
    labels_true_ans = ds["label_true_base"] == ans
    return labels_true_ans

# def ds2label_model_obey_adapt(ds):
#     """label: whether the model obeyed the instruction to lie"""
#     label_instructed = ds["label_true_adapt"] ^ ds["instructed_to_lie_adapt"]
#     ans = ds["binary_ans_adapt"] > 0.5
#     labels_untruth = label_instructed == ans
#     return labels_untruth

# def ds2label_model_truth_adapt(ds):
#     """label: whether the model told the truth"""
#     ans = ds["binary_ans_adapt"] > 0.5
#     labels_true_ans = ds["label_true_adapt"] == ans
#     return labels_true_ans

def ranking_truth_telling(ds):
    """proxy label whether the adapter or the base model were more truthful."""
    #note if we know which model is more truthful, and it's a binary choice, we can take the choice that is in that direction as the truth
    return ds['correct_truth_telling_base'] > ds['correct_truth_telling_adapt']


def undo_ranked_truth_telling(base_more_truthful: Float[Tensor, ''], ds: Dataset, ):
    # note if we know which model is more truthful, and it's a binary choice, we can take the choice that is in that direction as the truth
    base_more_positive = (ds['binary_ans_base'] > ds['binary_ans_adapt']) * 1.0
    return switch(base_more_positive, base_more_truthful)

def ranking_instruction_following(ds):
    """proxy label whether the adapter or the base model were more instruction following."""
    return ds['correct_instruction_following_base'] < ds['correct_instruction_following_adapt']



LABEL_MAPPING = dict(label_model_truth=ds2label_model_truth, label_model_obey=ds2label_model_obey,
                        # label_model_truth_adapt=ds2label_model_truth_adapt, label_model_obey_adapt=ds2label_model_obey_adapt,
                        ranking_truth_telling=ranking_truth_telling, ranking_instruction_following=ranking_instruction_following)
 
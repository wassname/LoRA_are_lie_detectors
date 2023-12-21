import gc
import torch
from datasets import Dataset
import numpy as np


# def ds_keep_cols(ds: Dataset, cols: list) -> Dataset:
#     cols_all = set(ds.features.keys())
#     cols_drop = cols_all - set(cols)
#     return ds.remove_columns(cols_drop)



def shuffle_dataset_by(ds, column):
    example_i = np.array(ds["example_i"])
    uniq_example_i = np.array(sorted(set(example_i)))
    shuffled_indices = np.random.permutation(uniq_example_i)
    index = np.arange(len(example_i))
    new_inds = np.concatenate([index[example_i == i] for i in shuffled_indices])
    return ds.select(new_inds)

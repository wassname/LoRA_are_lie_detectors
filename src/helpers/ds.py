import gc
import torch
from datasets import Dataset
import numpy as np
from random import Random

# def ds_keep_cols(ds: Dataset, cols: list) -> Dataset:
#     cols_all = set(ds.features.keys())
#     cols_drop = cols_all - set(cols)
#     return ds.remove_columns(cols_drop)



def shuffle_dataset_by(ds, column: str="example_i", rng: Random = Random(42)):
    example_i = np.array(ds[column])
    uniq_example_i = np.array(sorted(set(example_i)))
    rng.shuffle(uniq_example_i)
    index = np.arange(len(example_i))
    new_inds = np.concatenate([index[example_i == i] for i in uniq_example_i])
    return ds.select(new_inds)

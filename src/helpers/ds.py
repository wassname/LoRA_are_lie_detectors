import gc
import torch
from datasets import Dataset
import datasets
import numpy as np
from random import Random
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Union, List, Dict, Any, Optional, Tuple

# def ds_keep_cols(ds: Dataset, cols: list) -> Dataset:
# don't need, have ds.select_columns
#     cols_all = set(ds.features.keys())
#     cols_drop = cols_all - set(cols)
#     return ds.remove_columns(cols_drop)


# def shuffle_dataset_by(ds: Dataset, column: str="example_i", rng: Random = Random(42)):
#     example_i = np.array(ds[column])
#     uniq_example_i = np.array(sorted(set(example_i)))
#     rng.shuffle(uniq_example_i)
#     index = np.arange(len(example_i))
#     new_inds = np.concatenate([index[example_i == i] for i in uniq_example_i])
#     return ds.select(new_inds)


def shuffle_dataset_by(
    *args, **kwargs
):
    ds_train, ds_test = train_test_split_ds(*args, **kwargs)
    ds_out = datasets.concatenate_datasets([ds_train, ds_test])
    return ds_out


def train_test_split_ds(
    ds: Dataset,
    target: str = "label_true_base",
    stratify_columns: List[str] = ["sys_instr_name_base"],
    random_state=42, test_size=0.5, **kwargs
):
    df = pd.DataFrame(
        ds.select_columns([target] + stratify_columns).with_format("numpy")
    ).reset_index()
    splitter = StratifiedShuffleSplit(random_state=random_state, test_size=test_size, **kwargs)
    train_indices, test_indices = next(
        splitter.split(df, df[target], groups=df[stratify_columns])
    )
    ds_train = ds.select(train_indices)
    ds_test = ds.select(test_indices)
    return ds_train, ds_test

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

plt.style.use("ggplot")

from typing import Optional, List, Dict, Union
from jaxtyping import Float
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch import optim
from torch.utils.data import random_split, DataLoader, TensorDataset

from pathlib import Path
from einops import rearrange

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    LoftQConfig,
    IA3Config,
)

import datasets
from datasets import Dataset

from loguru import logger

logger.add(os.sys.stderr, format="{time} {level} {message}", level="INFO")


import lightning.pytorch as pl
from src.datasets.dm import DeceptionDataModule
from src.models.pl_lora_ft import AtapterFinetuner

from src.config import ExtractConfig
from src.prompts.prompt_loading import load_preproc_dataset
from src.models.load import load_model
from src.helpers.torch import clear_mem


# # quiet please
torch.set_float32_matmul_precision("medium")
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# params
max_epochs = 1
device = "cuda:0"

cfg = ExtractConfig(
    batch_size=3,
    max_examples=(200, 60),
)

checkpoint_path = "notebooks/lightning_logs/version_116/final"

model, tokenizer = model, tokenizer = load_model(
    cfg.model,
    device=device,
    adaptor_path=checkpoint_path,
    dtype=torch.float16,
)
clear_mem()

datasets2 = []
for ds_name in cfg.datasets_oos:
    N = sum(cfg.max_examples) // 2
    ds_tokens1 = load_preproc_dataset(
        ds_name,
        tokenizer,
        N=N,
        seed=cfg.seed,
        num_shots=cfg.num_shots,
        max_length=cfg.max_length,
        prompt_format=cfg.prompt_format,
    ).with_format("torch")
    datasets2.append(ds_tokens1)
ds_tokens2 = datasets.concatenate_datasets(datasets2)


dl_oos2 = DataLoader(
    ds_tokens2, batch_size=cfg.batch_size * 2, drop_last=False, shuffle=False
)
len(ds_tokens2)


from src.helpers.torch import clear_mem, detachcpu, recursive_copy
from src.models.pl_lora_ft import postprocess_result

from src.eval.collect import generate_batches, manual_collect2

# debugging
# o = next(iter(generate_batches(dl_oos2, model)))

ds_out, f = manual_collect2(dl_oos2, model, dataset_name="oos2")


from src.eval.helpers import test_intervention_quality2
from src.eval.labels import ds2label_model_obey, ds2label_model_truth


for label_name, label_fn in dict(
    label_model_truth=ds2label_model_truth, label_model_obey=ds2label_model_obey
).items():
    # fit probe
    print("=" * 80)
    print("making intervention with", label_name, "hidden states")
    test_intervention_quality2(ds_out, label_fn, tokenizer)

for label_name, label_fn in dict(
    label_model_truth=ds2label_model_truth, label_model_obey=ds2label_model_obey
).items():
    # fit probe
    print("=" * 80)
    print("making intervention with", label_name, "diff(hidden states)")
    test_intervention_quality2(ds_out, label_fn, take_diff=True)

from src.eval.ds import qc_ds

qc_ds(ds_out)

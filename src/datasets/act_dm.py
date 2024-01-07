from datasets import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from loguru import logger
import uuid
from src.helpers.shared_dataset import SharedDataset

def to_ds(ds, name):
    tds = torch.utils.data.TensorDataset(ds['X'], ds['y'])
    return SharedDataset(tds, name)

class ActivationDataModule(pl.LightningDataModule):
    """
    Pass in an on-disc dataset on hmodel activations. Should have only the columns you want to return as a dict (e.g. X and y). You can prepare the dataset as a streaming map like so

    ```py
    def ds2xy(row):
        X = torch.cat([row['end_residual_stream_base'], row['end_residual_stream_adapt']], dim=1)[:, SKIP::STRIDE]
        y = row['binary_ans_base']-row['binary_ans_adapt']
        return dict(X=X, y=y)

    def prepare_ds(ds):
        ds = ds.to_iterable_dataset().with_format("torch")
        ds = ds.map(ds2xy)
        ds = ds.select_columns(['X', 'y'])
        return ds
    ```
    """

    def __init__(
        self,
        ds: Dataset,
        name: str,
        num_workers: int = 0,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ds"])
        self.ds = ds.with_format("torch")
        # self.setup("train")
        self.hash = uuid.uuid4().hex[:6]

    def to_tds(self, ds, name):
        """huggingface dataset to pytorch."""
        h = self.hparams
        # 4x faster if we make it a tensor ourselves
        ds = ds.with_format(None)
        tds = torch.utils.data.TensorDataset(
            torch.FloatTensor(ds['X']), torch.FloatTensor(ds['y']))
        
        # this shared dataset is 10x faster with multiple workers
        if h.num_workers>0: 
            tds = SharedDataset(tds, f"{self.hparams.name}_{self.hash}_{name}") 
        return tds

    def setup(self, stage: str = "train"):
        n = len(self.ds)
        self.splits = {
            "train": (0, int(n * 0.5)),
            "val": (int(n * 0.5), int(n * 0.75)),
            "test": (int(n * 0.75), n),
        }
        logger.info(f'converting datasets this may take a while... {self.hparams.name} {stage}')
        if stage=="train":
            self.datasets = {
                key: self.to_tds(self.ds.select(range(start, end)), key)
                for key, (start, end) in self.splits.items()
            }
        elif stage=="all":
            self.datasets = {"all": to_ds(self.ds, 'all')}
        else:
            raise NotImplementedError(f"unknown stage {stage}")

    def create_dataloader(self, name, shuffle=False):
        h = self.hparams
        # 4x faster if we make it a tensor ourselves
        tds = self.datasets[name]
        return DataLoader(
            tds, batch_size=h.batch_size, drop_last=False, shuffle=shuffle, num_workers=h.num_workers,
            pin_memory=True,
            # timeout=20 if h.num_workers>0 else None, 
            # persistent_workers=h.num_workers>0,
        )

    def train_dataloader(self):
        return self.create_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader("val")

    def test_dataloader(self):
        return self.create_dataloader("test")
    
    def all_dataloader(self):
        return self.create_dataloader("all")



class TokenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ds: Dataset,
        # name: str,
        num_workers: int = 0,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ds"])
        self.ds = ds.with_format("torch")
        self.setup("train")


    def setup(self, stage: str = "train"):
        n = len(self.ds)
        self.splits = {
            "train": (0, int(n * 0.5)),
            "val": (int(n * 0.5), int(n * 0.75)),
            "test": (int(n * 0.75), n),
        }
        logger.info('converting datasets this may take a while...')
        if stage=="train":
            self.datasets = {
                key: self.ds.select(range(start, end))
                for key, (start, end) in self.splits.items()
            }
        elif stage=="all":
            self.datasets = {"all": self.ds}
        else:
            raise NotImplementedError(f"unknown stage {stage}")

    def create_dataloader(self, name, shuffle=False):
        h = self.hparams
        # 4x faster if we make it a tensor ourselves
        tds = self.datasets[name]
        return DataLoader(
            tds, batch_size=h.batch_size, drop_last=False, shuffle=shuffle, num_workers=h.num_workers,
        )

    def train_dataloader(self):
        return self.create_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader("val")

    def test_dataloader(self):
        return self.create_dataloader("test")
    
    def all_dataloader(self):
        return self.create_dataloader("all")

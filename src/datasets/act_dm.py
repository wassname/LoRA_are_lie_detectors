from datasets import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


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
        num_workers: int = 0,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ds"])
        self.ds = ds.with_format("torch")
        self.setup("train")

    def setup(self, stage: str):
        n = len(self.ds)
        self.splits = {
            "train": (0, int(n * 0.5)),
            "val": (int(n * 0.5), int(n * 0.75)),
            "test": (int(n * 0.75), n),
        }

        self.datasets = {
            key: self.ds.select(range(start, end))
            for key, (start, end) in self.splits.items()
        }

    def create_dataloader(self, ds, shuffle=False):
        h = self.hparams
        return DataLoader(
            ds, batch_size=h.batch_size, drop_last=False, shuffle=shuffle, num_workers=h.num_workers
        )

    def train_dataloader(self):
        return self.create_dataloader(self.datasets["train"], shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.datasets["val"])

    def test_dataloader(self):
        return self.create_dataloader(self.datasets["test"])
    
    def all_dataloader(self):
        return DataLoader(self.ds, batch_size=self.hparams.batch_size, drop_last=False, shuffle=False)

from datasets import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DeceptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ds: Dataset,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ds"])
        self.ds = ds.with_format("torch")
        # self.x_cols = x_cols
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
        return DataLoader(
            ds, batch_size=self.hparams.batch_size, drop_last=False, shuffle=shuffle
        )

    def train_dataloader(self):
        return self.create_dataloader(self.datasets["train"], shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.datasets["val"])

    def test_dataloader(self):
        return self.create_dataloader(self.datasets["test"])

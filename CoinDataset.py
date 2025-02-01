import json
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class CoinDataset(pl.LightningDataModule):

    def __init__(self, batch_size=64, root='coins', transform=None, no_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = os.path.join(os.getcwd(), 'data', root)
        self.transform = transform
        self.no_workers = no_workers
        with open(os.path.join(self.data_dir, 'cat_to_name.json')) as f:
            self.classes = json.load(f)

    def setup(self, stage: str = None):
        if stage in (None, "fit") or stage is None:
            self.train = ImageFolder(
                os.path.join(self.data_dir, 'train'),
                transform=self.transform
            )
            self.val = ImageFolder(
                os.path.join(self.data_dir, 'validation'),
                transform=self.transform
            )
        if stage in (None, "test") or stage is None:
            self.test = ImageFolder(
                os.path.join(self.data_dir, 'test'),
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            self.batch_size,
            shuffle=True,
            num_workers=self.no_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            self.batch_size,
            shuffle=True,
            num_workers=self.no_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            self.batch_size,
            num_workers=self.no_workers
        )

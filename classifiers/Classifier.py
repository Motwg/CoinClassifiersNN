import time

import pytorch_lightning as pl
import torchmetrics
from torch import optim
from torch.nn import functional as f


class Classifier(pl.LightningModule):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.epoch = 0.0
        self.start = None

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_start(self):
        self.start = time.time()

    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        predicted = self.forward(x)
        loss = f.cross_entropy(predicted, target)

        # log metrics
        self.train_accuracy(predicted, target)
        self.log('Accuracy', {'Train': self.train_accuracy}, on_epoch=True, on_step=False, prog_bar=True)
        self.log('Loss', {'Train': loss})
        return loss

    def on_train_epoch_start(self) -> None:
        self.epoch += 1
        self.log('epoch', self.epoch)

    def on_train_epoch_end(self):
        self.log('train_time', time.time() - self.start)

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        predicted = self.forward(x)
        loss = f.cross_entropy(predicted, target)

        # log metrics
        self.val_accuracy(predicted, target)
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        self.log('Accuracy', {'Val': self.val_accuracy}, on_epoch=True, on_step=False, prog_bar=True)
        self.log('Loss', {'Val': loss}, on_epoch=True, on_step=False)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, target = test_batch
        predicted = self.forward(x)
        loss = f.cross_entropy(predicted, target)

        # log metrics
        self.test_accuracy(predicted, target)
        self.log('Accuracy', {'Test': self.test_accuracy}, on_epoch=True, on_step=False, prog_bar=True)
        self.log('Loss', {'Test': loss})
        return loss

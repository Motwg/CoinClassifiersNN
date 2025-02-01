import os
import torch

# using only 50% of capabilities
if torch:
    torch.cuda.set_per_process_memory_fraction(0.5, 0)

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping,  ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms as t

from CoinDataset import CoinDataset
from classifiers.CoinClassifier import CoinClassifier

if __name__ == '__main__':
    experiment = 2
    device = 'gpu'
    no_workers = min(1, os.cpu_count())
    epoch, step = 67, 6867
    load = os.path.join('checkpoints', f'Coins_{experiment}',
                        'global', f'epoch={epoch}-step={step}.ckpt')

    transform = t.Compose([
        t.Resize(300),
        t.RandomRotation(90),
        t.ColorJitter(brightness=0.2, hue=0.1),
        t.RandomCrop(size=(224, 224)),
        t.ToTensor()
    ])

    transform_t2 = t.Compose([
        t.Resize(300),
        t.RandomRotation(30),
        t.ColorJitter(brightness=0.2, hue=0.1),
        t.RandomCrop(size=(224, 224)),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    checkpoint_val_cb = ModelCheckpoint(
        dirpath=f'checkpoints/Coins_{experiment}/val',
        save_top_k=3,
        monitor='val_acc',
        mode='max'
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=f'checkpoints/Coins_{experiment}/global',
        save_top_k=5,
        monitor='epoch',
        mode='max'
    )
    early_stop_cb = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
    )

    if experiment == 1:
        dm = CoinDataset(transform=transform, no_workers=no_workers)
        no_classes = len(dm.classes)
        model = CoinClassifier(learning_rate=1e-5, pretrained=False, no_classes=no_classes)
    elif experiment == 2:
        dm = CoinDataset(transform=transform_t2, no_workers=no_workers)
        no_classes = len(dm.classes)
        model = CoinClassifier(learning_rate=1e-4, pretrained=True, no_classes=no_classes)
    else:
        raise Exception(f'Invalid experiment: {experiment}')

    logger = TensorBoardLogger(save_dir='logs', name=f'Coins_{experiment}')
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator=device,
        devices=1,
        logger=logger,
        # auto_scale_batch_size='binsearch',
        callbacks=[
            # early_stop_cb,
            checkpoint_cb,
            checkpoint_val_cb
        ]
    )
    # dm.setup('fit')
    # inputs, classes = next(iter(dm.train_dataloader()))
    # print(classes)

    # trainer.tune(model)
    # try:
    #     trainer.fit(model, dm, ckpt_path=load)
    # except (PermissionError, FileNotFoundError):
    #     trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path=load)

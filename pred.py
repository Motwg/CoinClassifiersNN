import os
import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision import transforms as t

from CoinDataset import CoinDataset
from classifiers.CoinClassifier import CoinClassifier

if __name__ == '__main__':
    experiment = 2
    no_workers = min(1, os.cpu_count())
    epoch, step = 67, 6867
    load = os.path.join('checkpoints', f'Coins_{experiment}',
                        'global', f'epoch={epoch}-step={step}.ckpt')
    transform_t2 = t.Compose([
        t.Resize(300),
        t.RandomRotation(30),
        t.ColorJitter(brightness=0.2, hue=0.1),
        t.RandomCrop(size=(224, 224)),
        t.ToTensor(),
        # t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dm = CoinDataset(transform=transform_t2, no_workers=no_workers)
    no_classes = len(dm.classes)
    model = CoinClassifier.load_from_checkpoint(load, learning_rate=1e-4)

    dm.setup('test')
    inputs, classes = next(iter(dm.test_dataloader()))

    model.eval()

    with torch.no_grad():
        y = model(t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs))

    fig = plt.figure(figsize=(10, 3))
    rows, columns = 4, 5
    ints = sorted([str(i + 1) for i in range(211)])
    mapping = {i: ints[i] for i in range(211)}

    print(classes)

    plt.rcParams.update({'font.size': 6})
    for i in range(rows * columns):
        all_pred = y[i].numpy()
        pred = np.where(all_pred == np.amax(all_pred))[0][0]

        fig.add_subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.title(
            f'GT: {dm.classes[mapping[classes[i].item()]]}\n'
            f'pred: {dm.classes[mapping[pred]]}'
        )
        plt.imshow(t.ToPILImage()(inputs[i]))
    plt.show()

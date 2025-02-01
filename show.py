from matplotlib import pyplot as plt
from torchvision import transforms as t

from CoinDataset import CoinDataset

if __name__ == '__main__':
    transform_t2 = t.Compose([
        t.Resize(300),
        t.RandomRotation(30),
        t.ColorJitter(brightness=0.2, hue=0.1),
        t.RandomCrop(size=(224, 224)),
        t.ToTensor(),
        # t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dm = CoinDataset(transform=transform_t2)
    no_classes = len(dm.classes)

    dm.setup('fit')
    inputs, classes = next(iter(dm.test_dataloader()))

    fig = plt.figure(figsize=(20, 6))
    rows, columns = 2, 5
    ints = sorted([str(i + 1) for i in range(211)])
    mapping = {i: ints[i] for i in range(211)}

    plt.rcParams.update({'font.size': 10})
    assert isinstance(dm.classes, dict)
    for i in range(10):
        fig.add_subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.title(dm.classes[mapping[classes[i].item()]])
        plt.imshow(t.ToPILImage()(inputs[i]))
    plt.show()

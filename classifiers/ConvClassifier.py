from torch import nn

from mnist_objective.Lighting import Classifier


class ConvClassifier(Classifier):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.layers = nn.Sequential(
            # Tensor shape (?, 28, 28, 1)
            nn.Conv2d(1, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            # Tensor shape (?, 26, 26, 16)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Tensor shape (?, 13, 13, 16)
            nn.Conv2d(16, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            # Tensor shape (?, 11, 11, 16)
            nn.Flatten(),
            nn.Linear(11 * 11 * 16, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 10, bias=True),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)

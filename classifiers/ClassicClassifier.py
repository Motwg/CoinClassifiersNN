from torch import nn


from mnist_objective.Lighting import Classifier


class ClassicClassifier(Classifier):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

        self.layers = nn.Sequential(
            # Tensor shape (?, 28 * 2 * 2)
            nn.Linear(28 * 2 * 2, 24, bias=True),
            nn.Tanh(),
            nn.Linear(24, 24, bias=True),
            nn.Tanh(),
            nn.Linear(24, 10, bias=True),
            nn.Softmax()
            # Tensor shape (?, 10)
        )

    def forward(self, x):
        return self.layers(x)

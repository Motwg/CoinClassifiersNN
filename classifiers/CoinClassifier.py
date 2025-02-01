from torch import nn

import torchvision.models as models

from classifiers.Classifier import Classifier


class CoinClassifier(Classifier):

    def __init__(self, learning_rate, pretrained=False, no_classes=211, learning=True):
        super().__init__(learning_rate)
        self.layers = nn.Sequential(
            sq_net := models.squeezenet1_1(pretrained=pretrained)
        )
        sq_net.classifier[1] = nn.Conv2d(512, no_classes, kernel_size=(1, 1), stride=(1, 1))
        print(sq_net)
        if not learning:
            for name, param in sq_net.features.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.layers(x)

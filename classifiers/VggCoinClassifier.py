from torch import nn

import torchvision.models as models

from classifiers.Classifier import Classifier


class VggCoinClassifier(Classifier):

    def __init__(self, learning_rate, pretrained=False, no_classes=211):
        super().__init__(learning_rate)
        self.layers = nn.Sequential(
            vgg16 := models.vgg16(pretrained=pretrained)
        )
        vgg16.classifier[6] = nn.Linear(4096, no_classes)
        print(vgg16)
        if pretrained:
            for name, param in vgg16.features.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.layers(x)

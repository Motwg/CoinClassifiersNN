from torch import nn

import torchvision.models as models

from mnist_objective.Lighting import Classifier


class SqueezeNetClassifier(Classifier):

    def __init__(self, learning_rate, pretrained, no_off_layers=(0, 0)):
        super().__init__(learning_rate)
        self.layers = nn.Sequential(
            sq_net := models.squeezenet1_1(pretrained=pretrained),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )
        sq_net.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        print(sq_net)
        if pretrained:
            for name, param in sq_net.features.named_parameters():
                param.requires_grad = False
            # for i, child in enumerate(sq_net.children()):
            #     for name, param in child.named_parameters():
            #         print(f'layer {i}: {name}')
            #         if no_off_layers[0] <= i <= no_off_layers[1]:
            #             param.requires_grad = False

    def forward(self, x):
        return self.layers(x)

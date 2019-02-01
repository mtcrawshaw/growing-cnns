import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Model
"""

class CustomConvNet(nn.Module):

    def __init__(self, config, num_classes=1000, init_weights=True, batch_norm=True):
        super(CustomConvNet, self).__init__()

        self.features = CustomConvNet._make_features(config, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), # 512 -> 512 * 7 * 7 for imagenet
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096), 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_features(cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for l in cfg:
            layer = []

            if l[0] == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif l[0] == 'C':
                out_channels = l[1]
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if batch_norm:
                    layer += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                else:
                    layer += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_channels

            layers.append(nn.Sequential(*layer))

        return nn.Sequential(*layers)



import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

"""
    Model
"""

class CustomConvNet(nn.Module):

    def __init__(self, config, num_classes=1000, init_weights=True, batch_norm=True):
        super(CustomConvNet, self).__init__()

        self.features = make_features(config, batch_norm=batch_norm)
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

"""
    Produces a pytorch Module for a network architecture described by
    the argument cfg.
"""
def make_features(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for l in cfg:
        layer = None

        if l[0] == 'M': # Max pooling

            layer = nn.MaxPool2d(kernel_size=2, stride=2)

        elif l[0] == 'C': # Convolution
            
            out_channels = l[1]
            layer = make_layer_C(in_channels, out_channels, batch_norm)
            in_channels = out_channels

        elif l[0] == 'R_Basic': # Resnet basic

            out_channels = l[1]
            layer = make_layer_R_Basic(in_channels, out_channels, batch_norm)
            in_channels = out_channels

        elif l[0] == 'R_Bottleneck': # Resnet bottleneck

            out_channels = l[1]
            layer = make_layer_R_Bottleneck(in_channels, out_channels,
                        batch_norm)
            in_channels = out_channels

        layers.append(layer)

    return nn.Sequential(*layers)


"""
    Helper functions for make_features
"""
def make_layer_M():
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_layer_C(in_channels, out_channels, batch_norm):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    if batch_norm:
        layer_list = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layer_list = [conv2d, nn.ReLU(inplace=True)]

    return nn.Sequential(*layer_list)

def make_layer_R_Basic(in_channels, out_channels, batch_norm):
    downsample = None

    if out_channels != in_channels:
        downsample = nn.Sequential(nn.Conv2d(in_channels,
            out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

    return BasicBlock(in_channels, out_channels,
                downsample=downsample)

def make_layer_R_Bottleneck(in_channels, out_channels, batch_norm):
    downsample = None
    if out_channels != in_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, kernel_size=1,
                bias=False),
            nn.BatchNorm2d(out_channels))

    if out_channels % Bottleneck.expansion != 0:
        raise ValueError('Number of out_channels %d must be a' +
            'multiple of Bottleneck.expansion %d.' % (out_channels,
            Bottleneck.expansion))

    return Bottleneck(in_channels, out_channels //
                Bottleneck.expansion, downsample=downsample)


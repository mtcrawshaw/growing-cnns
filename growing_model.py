import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision.models.resnet import conv3x3, BasicBlock, Bottleneck

"""
    Growing VGG Model
"""

class GrowingVGG(nn.Module):

    def __init__(self, num_classes=1000, batch_norm=True):
        super(GrowingVGG, self).__init__()

        self.batch_norm = batch_norm

        # Define growth steps
        self.current_config = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']
        self.growth_steps = []
        self.growth_steps.append([(1, 64), (4, 128), (7, 256), (10, 512), (13, 512)])
        self.growth_steps.append([(8, 256), (12, 512), (16, 512)])
        
        self.current_step = 0
        self.features = make_layers(self.current_config, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), # 512 -> 512 * 7 * 7 for imagenet
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096), 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            m = GrowingVGG._initialize_layer(m)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def step(self):
        assert self.current_step < len(self.growth_steps)

        # Save current state dictionary
        state_dict = self.state_dict()

        # Create new layers
        old_config = list(self.current_config)
        self._grow_current_config()
        self.features = make_layers(self.current_config, batch_norm=self.batch_norm)

        # Rename keys in old_state_dict to match new layer names
        module_index = 0
        old_module_index = 0
        next_new_layer_index = 0
        next_new_layer_pos, next_new_layer = self.growth_steps[self.current_step][next_new_layer_index]
        for i in range(len(self.current_config)):
            current_layer = self.current_config[i]

            if i == next_new_layer_pos:
                next_new_layer_index += 1
                if next_new_layer_index < len(self.growth_steps[self.current_step]):
                    next_new_layer_pos, next_new_layer = self.growth_steps[self.current_step][next_new_layer_index]
            else:
                new_key = 'features.' + str(module_index) + '.'
                old_key = 'features.' + str(old_module_index) + '.'

                for param in ['weight', 'bias']:
                    state_dict[new_key + param] = state_dict.pop(old_key + param)

                if current_layer == 'M':
                    old_module_index += 1
                else:
                    old_module_index += 2

            if current_layer == 'M':
                module_index += 1
            else:
                module_index += 2
        
        self.load_state_dict(state_dict, strict=False)
        self.current_step += 1

    def _grow_current_config(self):
        new_config = list(self.current_config)

        for pos, layer in self.growth_steps[self.current_step]:
            new_config.insert(pos, layer)

        self.current_config = new_config
        self.current_step += 1

    @staticmethod
    def _initialize_layer(m):
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

        return m

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


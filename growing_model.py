import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision.models.resnet import conv3x3, BasicBlock, Bottleneck

from model import make_layers, VGG

"""
    Growing VGG Model
"""

class GrowingVGGController():

    def __init__(self, num_classes=1000, batch_norm=True):
        
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        # Define growth steps
        self.current_config = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']
        self.growth_steps = []
        self.growth_steps.append([(1, 64), (4, 128), (7, 256), (10, 512), (13, 512)])
        self.growth_steps.append([(8, 256), (12, 512), (16, 512)])
        
        self.current_step = -1

    def step(self, state_dict=None, parallel=True):
        """
        Produces the next step in the growing model from the previous state dictionary
        Args:
            state_dict (Ordered dict): State dictionary from model to be grown
            parallel (bool): Whether or not training is running on multiple GPUs, which affects parameter names in state_dict
        """
        assert self.current_step < len(self.growth_steps)

        # Initially creating model
        if self.current_step == -1:
            self.current_step = 0
            model = VGG(make_layers(self.current_config, batch_norm=self.batch_norm), num_classes=self.num_classes) 
            return model

        # Create new layers
        old_config = list(self.current_config)
        self._grow_current_config()
        model = VGG(make_layers(self.current_config, batch_norm=self.batch_norm), num_classes=self.num_classes)

        # Rename keys in old_state_dict to match new layer names
        module_index = 0
        old_module_index = 0
        next_new_layer_index = 0
        next_new_layer_pos, next_new_layer = self.growth_steps[self.current_step][next_new_layer_index]

        param_base = 'features.module.' if parallel else 'features.'

        for i in range(len(self.current_config)):
            current_layer = self.current_config[i]

            if i == next_new_layer_pos:
                next_new_layer_index += 1
                if next_new_layer_index < len(self.growth_steps[self.current_step]):
                    next_new_layer_pos, next_new_layer = self.growth_steps[self.current_step][next_new_layer_index]
            else:
                if current_layer == 'M':
                    old_module_index += 1
                else:
                    new_key = param_base + str(module_index) + '.'
                    old_key = param_base + str(old_module_index) + '.'

                    for param in ['weight', 'bias']:
                        state_dict[new_key + param] = state_dict.pop(old_key + param)

                    old_module_index += 3 if self.batch_norm else 2

            if current_layer == 'M':
                module_index += 1 # Max pooling layer only contains 1 module (a max pooling module)
            else:
                module_index += 3 if self.batch_norm else 2 # Convolutional layer contains 2 modules (conv and
                                                            # relu), 3 with batch norm
        
        # Load newly named parameters from state_dict into new model
        model.load_state_dict(state_dict, strict=False)
        self.current_step += 1

        return model

    def _grow_current_config(self):
        new_config = list(self.current_config)

        for pos, layer in self.growth_steps[self.current_step]:
            new_config.insert(pos, layer)

        self.current_config = new_config


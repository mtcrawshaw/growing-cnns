import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision.models.resnet import conv3x3, BasicBlock, Bottleneck

from model import CustomConvNet

"""
    Growth Controller
"""

class GrowthController():

    def __init__(self, initial_config, growth_steps, num_classes=1000, batch_norm=True):
        
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        # Define growth steps
        self.current_config = list(initial_config)
        self.growth_steps = list(growth_steps)
        self.current_step = -1

        # Define growth history, a list of the growth steps each layer was
        # inserted during. So on step 0, this list will be all zeroes and with
        # length equal to the number of layers on iteration 0. On step 1, this
        # list will contain mostly zeroes, with ones inserted at the positions
        # where new layers were inserted during the growth step, etc.
        self.growth_history = []

    def step(self, old_model=None):
        """
        Produces the next step in the growing model from the previous state dictionary
        Args:
            current_model (CustomConvNet): Model to be grown, that was also returned from step
        """
        assert self.current_step < len(self.growth_steps)

        # Initially creating model
        if self.current_step == -1:
            self.current_step = 0
            new_model = CustomConvNet(self.current_config, num_classes=self.num_classes, batch_norm = self.batch_norm) 
            self.growth_history = [0 for i in range(len(self.current_config))]
            return new_model

        # Create new layers
        old_config = list(self.current_config)
        self._grow_current_config()
        new_model = CustomConvNet(self.current_config, num_classes=self.num_classes, batch_norm = self.batch_norm)

        # Transfer weights from old model to new model
        old_layer_index = 0
        next_new_layer_index = 0
        next_new_layer_pos = self.growth_steps[self.current_step][next_new_layer_index][0]
        next_new_layer = self.growth_steps[self.current_step][next_new_layer_index][1:] 

        for new_layer_index in range(len(self.current_config)):
            current_layer = self.current_config[new_layer_index]

            if new_layer_index == next_new_layer_pos: # Current layer is newly inserted in growth step
                next_new_layer_index += 1
                if next_new_layer_index < len(self.growth_steps[self.current_step]):
                    next_new_layer_pos = self.growth_steps[self.current_step][next_new_layer_index][0]
                    next_new_layer = self.growth_steps[self.current_step][next_new_layer_index][1:]
                
                self.growth_history.insert(new_layer_index, self.current_step +
                        1)

            else:                       # Current layer is an old layer
                if current_layer[0] == 'C':
                    state_dict = old_model.features.__getitem__(old_layer_index).state_dict()
                    new_model.features.__getitem__(new_layer_index).load_state_dict(state_dict)

                old_layer_index += 1
        
        # Transfer classifier weights
        new_model.classifier.load_state_dict(old_model.classifier.state_dict())

        self.current_step += 1
        return new_model

    """
    Changes self.current_config to reflect one growth step based on
    self.growth_steps.
    """
    def _grow_current_config(self):
        new_config = list(self.current_config)

        for step in self.growth_steps[self.current_step]:
            pos = step[0]
            layer = step[1:]
            new_config.insert(pos, layer)

        self.current_config = new_config


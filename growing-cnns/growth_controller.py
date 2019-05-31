import math

from model import CustomConvNet

"""
    Growth Controller
"""

class GrowthController():

    def __init__(self, growth_steps, num_classes=1000, batch_norm=True):
        
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.growth_steps = growth_steps
        self.current_step = -1

        self.initial_channels = 4
        self.max_pools = 3
        self.conv_per_max_pool = 1

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
        assert self.current_step < self.growth_steps

        # Initially creating model
        if self.current_step == -1:
            self.current_step = 0
            new_model = CustomConvNet(self.initial_channels, self.max_pools,
                    self.conv_per_max_pool, num_classes=self.num_classes,
                    batch_norm=self.batch_norm) 
            num_layers = self.max_pools * (self.conv_per_max_pool + 1)
            self.growth_history = [0 for i in range(num_layers)]
            return new_model

        # Create new model
        self.current_step += 1
        self.conv_per_max_pool += 1
        new_model = CustomConvNet(self.initial_channels, self.max_pools,
                self.conv_per_max_pool, num_classes=self.num_classes,
                batch_norm = self.batch_norm)

        # Transfer weights from old model to new model
        for i in range(self.max_pools):

            """
            Here we transfer weights into all layers of the new model between
            the ith and (i + 1)th max pooling layer, other than the second to
            last layer. The second to last layer will be initialized to
            calculate the identity function, and it serves as the newly grown
            layer. The reason that the second to last layer is the new one,
            as opposed to the last layer, is because the last layer doubles
            the number of channels (before the width and height are halved by
            the max pool) so a new layer cannot be placed after the last layer.
            """
            new_layer_pos = self.conv_per_max_pool - 2
            for j in range(self.conv_per_max_pool - 1):
                if j == new_layer_pos:
                    continue

                # Grab state dictionary from old layer
                old_layer_index = self.conv_per_max_pool * i + j
                old_layer = old_model.features.__getitem__(old_layer_index)
                old_state_dict = old_layer.state_dict()

                # Load old state dictionary into new layer
                new_layer_index = (self.conv_per_max_pool + 1) * i + j
                new_layer = new_model.features.__getitem__(new_layer_index)
                new_layer.load_state_dict(old_state_dict)

            self.growth_history.insert(i * self.conv_per_max_pool,
                    self.current_step)

        # Transfer classifier weights
        new_model.classifier.load_state_dict(old_model.classifier.state_dict())

        return new_model

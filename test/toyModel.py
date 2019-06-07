import torch
import torch.nn as nn

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        self.features = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self._initializeWeights()

    def forward(self, x):
        return self.features(x)

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


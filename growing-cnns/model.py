import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

class CustomConvNet(nn.Module):

    def __init__(self, initialChannels=64, maxPools=4, convPerMaxPool=3,
            numClasses=1000, randomWeights=True, batchNorm=True,
            classifierHiddenSize=2048):
        super(CustomConvNet, self).__init__()

        self.features = makeFeatures(initialChannels, maxPools,
                convPerMaxPool, batchNorm=batchNorm)

        # Calculate size of self.features output
        # This setting of hardcoding width and height is temporary, it should
        # be made dependent on the dataset.
        width = IMAGE_WIDTH / (2 ** maxPools)
        height = IMAGE_HEIGHT / (2 ** maxPools)
        depth = initialChannels * (2 ** maxPools)
        outputSize = int(width * height * depth)

        self.classifier = nn.Sequential(
            nn.Linear(outputSize, classifierHiddenSize), 
            nn.ReLU(True),
            nn.Linear(classifierHiddenSize, classifierHiddenSize), 
            nn.ReLU(True),
            nn.Linear(classifierHiddenSize, numClasses),
        )

        self._initializeWeights(randomWeights)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    """
        If randomWeights is False, then the weights of the convolutional
        layers are initialized with the Dirac delta function, so that
        each convolutional layer calculates the identity function.
    """
    def _initializeWeights(self, randomWeights=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if randomWeights:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.dirac_(m.weight)

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
def makeFeatures(initialChannels, maxPools, convPerMaxPool,
        batchNorm=True):
    layers = []
    inChannels = 3

    # Build up list of layers
    outChannels = initialChannels
    for i in range(maxPools):

        # Convolutional layers between max pools
        for j in range(convPerMaxPool):
            singleLayer = []

            # Convolution (double number of channels before max pool)
            if j == convPerMaxPool - 1:
                outChannels *= 2
            conv2d = nn.Conv2d(inChannels, outChannels, kernel_size=3,
                    padding=1)
            singleLayer.append(conv2d)

            # Batch normalization
            if batchNorm:
                singleLayer.append(nn.BatchNorm2d(outChannels))

            # Relu
            singleLayer.append(nn.ReLU(inplace=True))

            # Add layer to list of layers
            singleLayer = nn.Sequential(*singleLayer)
            layers.append(singleLayer)

            inChannels = outChannels

        # Max pooling layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


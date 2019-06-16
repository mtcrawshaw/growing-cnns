import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from listModule import ListModule

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

class CustomConvNet(nn.Module):

    def __init__(self, edges, convPerSection=3, initialChannels=64, maxPools=4,
            numClasses=1000, randomWeights=True, batchNorm=True,
            classifierHiddenSize=2048):
        super(CustomConvNet, self).__init__()

        # Parse edge list into adjacency list
        self.edges = list(edges)
        self.adjList = [[] for _ in range(convPerSection)]
        self.revAdjList = [[] for _ in range(convPerSection)]
        self.inDegree = [0 for _ in range(convPerSection)]
        self.outDegree = [0 for _ in range(convPerSection)]

        for s, e in edges:
            self.inDegree[e] += 1
            self.outDegree[s] += 1
            self.adjList[s].append(e)
            self.revAdjList[e].append(s)

        # Node 0 is the input node, the last node in the list is the output
        assert self.inDegree[0] == 0
        assert self.outDegree[-1] == 0

        # So that input to first node is input to forward (see sectionForward)
        self.revAdjList[0].append(-1)

        # Create sections
        self.convPerSection = convPerSection
        self.sections = makeSections(initialChannels, maxPools,
                convPerSection, batchNorm=batchNorm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate size of output to classifier
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
        
        x = convForward(self, x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def convForward(self, x):

        for section in self.sections:
            x = self.sectionForward(section, x)
            x = self.pool(x)

        return x

    def sectionForward(self, section, x):
        
        outputs = [None for _ in range(self.convPerSection)] + [x]
        queue = deque([0])
        inDegree = self.inDegree.copy()

        while queue:
            now = queue.popleft()
            inputList = [outputs[i] for i in self.revAdjList[now]]
            feed = torch.mean(torch.stack(inputList), dim=0)
            outputs[now] = section[now](feed)
            for v in self.adjList[now]:
                inDegree[v] -= 1
                if inDegree[v] == 0:
                    queue.append(v)

        # Returning output of last node
        return outputs[-2]

    """
        If randomWeights is False, then the weights of the convolutional
        layers are initialized with the Dirac delta function, so that
        each convolutional layer calculates the identity function.
    """
    def _initializeWeights(self, randomWeights):
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
    Produces a list of lists of modules, where each sublist corresponds to
    a single section of the network architecture (a section is a sequence of
    layers between two max pooling layers). 
"""
def makeSections(initialChannels, maxPools, convPerMaxPool,
        batchNorm=True):

    sections = []
    inChannels = 3

    # Build up list of layers
    outChannels = initialChannels
    for i in range(maxPools):

        sections.append([])

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
            sections[-1].append(singleLayer)

            inChannels = outChannels

        sections[-1] = ListModule(*sections[-1])

    return ListModule(*sections)


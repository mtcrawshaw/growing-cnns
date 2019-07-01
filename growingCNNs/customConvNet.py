import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# This is in case no parent packages are imported, such as in the test cases
try:
    from .utils.listModule import ListModule
except:
    from utils.listModule import ListModule

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

class CustomConvNet(nn.Module):

    def __init__(self, compGraph, initialChannels=64, numSections=4,
            numClasses=1000, randomWeights=True, batchNorm=True,
            joinWeighting='uniform'):

        super(CustomConvNet, self).__init__()

        # Parse edge list into adjacency list
        self.compGraph = compGraph.clone()
        self.adjList = [[] for _ in range(self.compGraph.numNodes)]
        self.revAdjList = [[] for _ in range(self.compGraph.numNodes)]
        self.inDegree = [0 for _ in range(self.compGraph.numNodes)]
        self.outDegree = [0 for _ in range(self.compGraph.numNodes)]

        for s, e in self.compGraph.edges:
            self.inDegree[e] += 1
            self.outDegree[s] += 1
            self.adjList[s].append(e)
            self.revAdjList[e].append(s)

        # Check that input node has no parent nodes and output node has no
        # child nodes
        inputIndex = self.compGraph.inputIndex
        outputIndex = self.compGraph.outputIndex
        assert self.inDegree[inputIndex] == 0
        assert self.outDegree[outputIndex] == 0

        # So that input to first node is input to forward (see sectionForward)
        self.revAdjList[0].append(-1)

        # Create sections
        self.initialChannels = initialChannels
        self.numSections = numSections
        self.batchNorm = batchNorm
        self.joinWeighting = joinWeighting
        self.sections = self.makeSections()
        self.joinWeights = self.makeJoinWeights()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate size of output to classifier
        # This setting of hardcoding width and height is temporary, it should
        # be made dependent on the dataset.
        width = int(IMAGE_WIDTH / (2 ** numSections))
        height = int(IMAGE_HEIGHT / (2 ** numSections))
        depth = int(initialChannels * (2 ** numSections))
        outputSize = width * height * depth

        # Create classifier
        self.avgPool = nn.AvgPool2d(width)
        self.classifier = nn.Linear(depth, numClasses)

        # Initialize weights
        self._initializeWeights(randomWeights)


    def forward(self, x):
        
        x = self.convForward(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def convForward(self, x):

        for section in self.sections:
            x = self.sectionForward(section, x)
            x = self.pool(x)

        return x

    def sectionForward(self, section, x):
        
        outputs = [None for _ in range(self.compGraph.numNodes)] + [x]
        inputIndex = self.compGraph.inputIndex
        outputIndex = self.compGraph.outputIndex
        queue = deque([inputIndex])
        inDegree = self.inDegree.copy()

        while queue:
            now = queue.popleft()
            inputList = [outputs[i] for i in self.revAdjList[now]]
            feed = self.joinInputs(section, now, inputList)
            outputs[now] = section[now](feed)
            for v in self.adjList[now]:
                inDegree[v] -= 1
                if inDegree[v] == 0:
                    queue.append(v)

        # Returning output of last node
        return outputs[outputIndex]

    """
        Defines how multiple inputs (convolution activations) are combined
        into a single tensor to be used as input to a given node's convolution
        operator. Modes are 'uniform', 'softmax', and 'free', where 'uniform'
        averages the inputs, 'softmax' performs a weighted average where the
        weight parameters are learned through backpropagation, and 'free'
        performs a weighted sum where the weight parameters are learned through
        backpropagation.
    """
    def joinInputs(self, section, node, inputList)

        assert self.joinWeighting in ['uniform', 'softmax', 'free']

        if self.joinWeighting == 'uniform':

            joined = sum([self.joinWeights[section][node][i] * inputList[i] for
                    i in range(len(inputList))])

        elif self.joinWeighting == 'softmax':

            normalizedWeights = None # Softmax over weights
            joined = sum([normalizedWeights[i] * inputList[i] for
                    i in range(len(inputList))])

        elif self.joinWeighting == 'free':

            joined = sum([self.joinWeights[section][node][i] * inputList[i] for
                    i in range(len(inputList))])

        return joined

    """
        If randomWeights is False, then the weights of the convolutional
        layers are initialized with the Dirac delta function, so that
        each convolutional layer calculates the identity function.
    """
    def _initializeWeights(self, randomWeights):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if randomWeights:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                            nonlinearity='relu')
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
    def makeSections(self):

        # Build up list of layers
        sections = []
        inChannels = 3
        outChannels = self.initialChannels

        for i in range(self.numSections):

            sections.append([])

            # Convolutional layers between max pools
            for j in range(self.compGraph.numNodes):
                singleLayer = []

                # Convolution (double number of channels before max pool)
                channels = outChannels
                if j == self.compGraph.outputIndex:
                    channels *= 2
                conv2d = nn.Conv2d(inChannels, channels, kernel_size=3,
                        padding=1)
                singleLayer.append(conv2d)

                # Batch normalization
                if self.batchNorm:
                    singleLayer.append(nn.BatchNorm2d(channels))

                # Relu
                singleLayer.append(nn.ReLU(inplace=True))

                # Add layer to list of layers
                singleLayer = nn.Sequential(*singleLayer)
                sections[-1].append(singleLayer)

                inChannels = outChannels

            inChannels *= 2
            outChannels *= 2

            sections[-1] = ListModule(*sections[-1])

        return ListModule(*sections)

    """
        Produces a list of lists of lists of constant variables, where
        joinWeights[i][j][k] represents the weight on the k-th input to the
        j-th node of the i-th section of the network.
    """
    def makeJoinWeights(self):

        networkList = []
        for section in range(self.numSections):

            sectionList = []
            for node in range(self.compGraph.numNodes):

                nodeList = []
                for input in range(self.inDegree[node]):

                    # The weights for 'softmax' and 'free' are initialized
                    # so that they calculate a weighted average, then allowed
                    # to change from there
                    if self.joinWeighting == 'uniform':
                        value = 1. / float(self.inDegree[node])
                        trainable = False
                    elif self.joinWeighting == 'softmax':
                        value = 1.
                        trainable = True
                    elif self.joinWeighting = 'free':
                        value = 1. / float(self.inDegree[node])
                        trainable = True

                    # None needs to be changed to a pytorch constant variable
                    # using value and trainable
                    nodeList.append(None)

                sectionList.append(ListModule(*nodeList))

            networkList.append(ListModule(*sectionList))

        joinWeights = ListModule(*networkList)
        return joinWeights

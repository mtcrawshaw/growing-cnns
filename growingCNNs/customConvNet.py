import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import ParameterList

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

        # Sort adjacency lists so that when inputs to a node are joined
        # (in function joinInputs), the index of a join weight is consistent
        # between growth steps
        for n in self.compGraph.nodes:
            self.adjList[n] = sorted(self.adjList[n])
            self.revAdjList[n] = sorted(self.revAdjList[n])

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

        for i, section in enumerate(self.sections):
            x = self.sectionForward(i, section, x)
            x = self.pool(x)

        return x

    def sectionForward(self, sectionIndex, section, x):
        
        outputs = [None for _ in range(self.compGraph.numNodes)] + [x]
        inputIndex = self.compGraph.inputIndex
        outputIndex = self.compGraph.outputIndex
        queue = deque([inputIndex])
        inDegree = self.inDegree.copy()

        while queue:
            now = queue.popleft()
            inputList = [outputs[i] for i in self.revAdjList[now]]
            inputs = torch.stack(inputList)
            feed = self.joinInputs(sectionIndex, now, inputs)
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
    def joinInputs(self, sectionIndex, node, inputs):

        assert self.joinWeighting in ['uniform', 'softmax', 'free']

        if self.joinWeighting == 'uniform':

            weights = self.joinWeights[sectionIndex][node]

        elif self.joinWeighting == 'softmax':

            weights = self.joinWeights[sectionIndex][node]
            weights = F.softmax(weights, dim=0)

        elif self.joinWeighting == 'free':

            weights = self.joinWeights[sectionIndex][node]

        weightedInputs = torch.mul(inputs, weights)
        weightedSum = torch.sum(weightedInputs, dim=0)
        return weightedSum

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

            sections[-1] = ModuleList(sections[-1])

        return ModuleList(sections)

    """
        Produces a list of lists of tensors, where joinWeights[i][j] represents
        the weights to the j-th node of the i-th section of the network.
    """
    def makeJoinWeights(self):

        sectionList = []
        for section in range(self.numSections):

            nodeList = []
            for node in range(self.compGraph.numNodes):

                numInputs = float(self.inDegree[node])
                numInputs = max(numInputs, 1.) # Avoid division by 0

                # The weights for 'softmax' and 'free' are initialized
                # so that they calculate a weighted average, then allowed
                # to change from there
                if self.joinWeighting == 'uniform':
                    value = 1. / numInputs
                    train = False
                elif self.joinWeighting == 'softmax':
                    value = 1.
                    train = True
                elif self.joinWeighting == 'free':
                    value = 1. / numInputs
                    train = True

                # Create weight tensor
                weightList = [value for i in range(int(numInputs))]
                weightTensor = torch.tensor(weightList, requires_grad=train)

                # Expand weight tensor to match input dimensions
                weightTensor = weightTensor.view(-1, 1, 1, 1, 1)
                nodeList.append(torch.nn.Parameter(weightTensor))

            sectionList.append(ParameterList(nodeList))

        joinWeights = ModuleList(sectionList)
        return joinWeights

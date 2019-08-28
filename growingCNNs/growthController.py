import math

# This is in case no parent packages are imported, such as in the test cases
try:
    from .customConvNet import CustomConvNet
    from .graphOperations import getInitialCompGraph, growCompGraph
except:
    from customConvNet import CustomConvNet
    from graphOperations import getInitialCompGraph, growCompGraph

"""
    Growth Controller
"""

class GrowthController():

    def __init__(self, initialChannels=64, numSections=4, initialNumNodes=3,
            growthSteps=3, numClasses=1000, batchNorm=True,
            growthMode='expandEdge', numConvToAdd=1, itemsToExpand='youngest',
            randomWeights=False, copyBatchNorm=True, joinWeighting='uniform'):
        
        self.numClasses = numClasses
        self.batchNorm = batchNorm
        self.growthSteps = growthSteps
        self.currentStep = -1

        self.initialChannels = initialChannels
        self.numSections = numSections
        self.numNodes = initialNumNodes
        self.growthMode = growthMode
        self.numConvToAdd = numConvToAdd
        self.itemsToExpand = itemsToExpand
        self.randomWeights = randomWeights
        self.copyBatchNorm = copyBatchNorm
        self.joinWeighting = joinWeighting

        # The ith element of growthHistory is the growth step during which the
        # ith node of the current model's computation graph was inserted.
        self.growthHistory = {}

    def step(self, oldModel=None):
        """
        Produces the next step in the growing model from the previous state dictionary
        Args:
            oldModel (CustomConvNet): Model to be grown, that was also returned from step
        """
        assert self.currentStep < self.growthSteps

        # Initially creating model
        if self.currentStep == -1:
            self.currentStep = 0
            compGraph = getInitialCompGraph(self.numNodes)
            newModel = CustomConvNet(
                    compGraph=compGraph,
                    initialChannels=self.initialChannels,
                    numSections=self.numSections,
                    numClasses=self.numClasses,
                    batchNorm=self.batchNorm,
                    joinWeighting=self.joinWeighting
            ) 
            for i in compGraph.nodes:
                self.growthHistory[i] = 0
            return newModel

        # Create new model
        self.currentStep += 1
        newCompGraph, nodesToCopy = growCompGraph(
                oldModel.compGraph,
                growthHistory=self.growthHistory,
                growthMode=self.growthMode,
                numConvToAdd=self.numConvToAdd,
                itemsToExpand=self.itemsToExpand
        )
        self.numNodes = newCompGraph.numNodes
        newModel = CustomConvNet(
                compGraph=newCompGraph,
                initialChannels=self.initialChannels,
                numSections=self.numSections,
                numClasses=self.numClasses,
                batchNorm=self.batchNorm, 
                randomWeights=self.randomWeights,
                joinWeighting=self.joinWeighting
        )

        # Transfer weights from old model to new model
        oldNodes = oldModel.compGraph.nodes
        newNodes = newModel.compGraph.nodes
        for i in range(self.numSections):

            # Copy weights from old model into nodes from new model with
            # same index.
            for j in newNodes:

                if j in oldNodes:

                    # Convolutional weights
                    oldLayer = oldModel.sections[i][j]
                    oldStateDict = oldLayer.state_dict()
                    newLayer = newModel.sections[i][j]
                    newLayer.load_state_dict(oldStateDict)

                    # Join weights
                    oldJoinWeights = oldModel.joinWeights[i][j]
                    n = oldJoinWeights.shape[0]
                    newModel.joinWeights[i][j][:n].data = oldJoinWeights.data

                else:

                    # Update growth history
                    self.growthHistory[j] = self.currentStep

            # Copy weights from old model into nodes from new model necessary
            # to preserve the function calculated by the network
            for sourceNode, destNode, weightToCopy in nodesToCopy:

                assert weightToCopy in ['conv', 'bn', 'join']

                if weightToCopy == 'conv':

                    # Grab state dictionary from source layer
                    sourceLayer = oldModel.sections[i][sourceNode]
                    sourceStateDict = sourceLayer.state_dict()

                    # Load source state dictionary into destination layer
                    destLayer = newModel.sections[i][destNode]
                    destLayer.load_state_dict(sourceStateDict)

                elif weightToCopy == 'bn':

                    if self.batchNorm and self.copyBatchNorm:

                        # Grab state dictionary from source layer
                        bnIndex = 1
                        sourceLayer = oldModel.sections[i][sourceNode][bnIndex]
                        sourceStateDict = sourceLayer.state_dict()

                        # Load source state dictionary into destination layer
                        destLayer = newModel.sections[i][destNode][bnIndex]
                        destLayer.load_state_dict(sourceStateDict)

                elif weightToCopy == 'join':

                    # Grab join weights from source node and insert into
                    joinWeights = oldModel.joinWeights[i][sourceNode].data
                    n = oldJoinWeights.shape[0]
                    newModel.joinWeights[i][destNode][:n].data = joinWeights


        # Transfer classifier weights
        newModel.classifier.load_state_dict(oldModel.classifier.state_dict())

        return newModel


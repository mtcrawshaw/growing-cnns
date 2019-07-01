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
            randomWeights=False):
        
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
                    batchNorm=self.batchNorm
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
                randomWeights=self.randomWeights
        )

        # Transfer weights from old model to new model
        oldNodes = oldModel.compGraph.nodes
        newNodes = newModel.compGraph.nodes
        for i in range(self.numSections):
            for j in newNodes:

                if j in oldNodes:

                    # Grab state dictionary from old layer
                    oldLayer = oldModel.sections[i][j]
                    oldStateDict = oldLayer.state_dict()

                    # Load old state dictionary into new layer
                    newLayer = newModel.sections[i][j]
                    newLayer.load_state_dict(oldStateDict)

                else:

                    # Update growth history
                    self.growthHistory[j] = self.currentStep

            for sourceNode, destNode in nodesToCopy:

                # Grab state dictionary from source layer
                sourceLayer = oldModel.sections[i][sourceNode]
                sourceStateDict = sourceLayer.state_dict()

                # Load source state dictionary into destination layer
                destLayer = newModel.sections[i][destNode]
                destLayer.load_state_dict(sourceStateDict)


        # Transfer classifier weights
        newModel.classifier.load_state_dict(oldModel.classifier.state_dict())

        return newModel

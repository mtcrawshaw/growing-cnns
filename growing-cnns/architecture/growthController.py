import math

from model import CustomConvNet
from computationGraph import ComputationGraph

"""
    Growth Controller
"""

class GrowthController():

    def __init__(self, initialChannels=64, maxPools=4, initialNumNodes=3,
            growthSteps=3, numClasses=1000, batchNorm=True,
            classifierHiddenSize=2048):
        
        self.numClasses = numClasses
        self.batchNorm = batchNorm
        self.growthSteps = growthSteps
        self.currentStep = -1

        self.initialChannels = initialChannels
        self.maxPools = maxPools
        self.numNodes = initialNumNodes
        self.classifierHiddenSize = classifierHiddenSize

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
            compGraph = self.getInitialCompGraph()
            newModel = CustomConvNet(
                    compGraph=compGraph,
                    initialChannels=self.initialChannels,
                    maxPools=self.maxPools,
                    numClasses=self.numClasses,
                    batchNorm=self.batchNorm,
                    classifierHiddenSize=self.classifierHiddenSize
            ) 
            for i in compGraph.nodes:
                self.growthHistory[i] = 0
            return newModel

        # Create new model
        self.currentStep += 1
        newCompGraph = self.growCompGraph(oldModel.compGraph)
        self.numNodes = newCompGraph.numNodes
        newModel = CustomConvNet(
                compGraph=newCompGraph,
                initialChannels=self.initialChannels,
                maxPools=self.maxPools,
                numClasses=self.numClasses,
                batchNorm=self.batchNorm, 
                classifierHiddenSize=self.classifierHiddenSize,
                randomWeights=False
        )

        # Transfer weights from old model to new model
        oldNodes = oldModel.compGraph.nodes
        newNodes = newModel.compGraph.nodes
        for i in range(self.maxPools):
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


        # Transfer classifier weights
        newModel.classifier.load_state_dict(oldModel.classifier.state_dict())

        return newModel

    def getInitialCompGraph(self):

        edges = [(i, i + 1) for i in range(self.numNodes - 1)]
        inputIndex = 0
        outputIndex = self.numNodes - 1
        return ComputationGraph(edges, inputIndex, outputIndex)

    def growCompGraph(self, compGraph):

        # Find nodes in current computation graph
        nodes = []
        for start, end in compGraph.edges:
            for node in start, end:
                if node not in nodes:
                    nodes.append(node)

        # Find next available node
        nextAvailableNode = 0
        while nextAvailableNode in nodes:
            nextAvailableNode += 1

        # Find nodes to expand
        nodesToExpand = list(compGraph.nodes)
        nodesToKeep = [compGraph.inputIndex, compGraph.outputIndex]
        for node in nodesToKeep:
            if node in nodesToExpand:
                nodesToExpand.remove(node)

        # Expand nodes
        newEdges = list(compGraph.edges)
        for node in nodesToExpand:
            
            # Expand node
            newNode = nextAvailableNode
            nextAvailableNode += 1

            tempEdges = []
            for start, end in newEdges:

                if node == start:
                    tempEdges.append((start, newNode))
                    tempEdges.append((newNode, end))
                else:
                    tempEdges.append((start, end))

            newEdges = list(tempEdges)

        newCompGraph = ComputationGraph(
                newEdges,
                compGraph.inputIndex,
                compGraph.outputIndex
        )
        return newCompGraph

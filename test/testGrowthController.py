import sys
import unittest
import importlib

import torch
import numpy as np

from testUtils import getTestInput
from toyModel import SimpleNet

sys.path.append('../growingCNNs/')
growthController = importlib.import_module('growthController')
GrowthController = growthController.GrowthController

class TestGrowthController(unittest.TestCase):

    # Helper function to test if weights in two different classifiers
    # have the same values
    def assertEqualClassifier(self, stateDict1, stateDict2):

        # Collect classifier parameter names
        classifierParams = [None, None]
        stateDicts = [stateDict1, stateDict2]
        for i, stateDict in enumerate(stateDicts):
            classifierParams[i] = [key for key in stateDict.keys() if
                    'classifier' in key]

        # Compare classifier parameter names
        for i in range(2):
            classifierParams[i] = set(classifierParams[i])
        difference = classifierParams[0] ^ classifierParams[1]
        self.assertTrue(len(difference) == 0)

        # Compare classifier parameter values
        classifierParams = classifierParams[0]
        for key in classifierParams:
            values = [stateDicts[i][key].numpy() for i in
                    range(len(stateDicts))]
            self.assertTrue(np.array_equal(values[0], values[1]))


    # Helper function to test if weights were properly transferred between
    # a network and its successor
    def assertEqualLayers(self, stateDict1, stateDict2, oldNodes, numSections):

        stateDicts = [stateDict1, stateDict2]

        # Check each layer that should've been transferred
        for section in range(numSections):

            # Collect layer parameter names
            paramNames = [None, None]
            for i, stateDict in enumerate(stateDicts):
                for node in oldNodes:
                    prefix = 'sections.%d.%d' % (section, node)
                    paramNames[i] = [key for key in stateDict.keys() if 
                            prefix in key]

            # Compare layer parameter names
            for i in range(2):
                paramNames[i] = set(paramNames[i])
            difference = paramNames[0] ^ paramNames[1]
            self.assertTrue(len(difference) == 0)
            paramNames = paramNames[0]

            # Compare layer parameter values
            for paramName in paramNames:

                # Collect and compare values
                values = [stateDicts[i][paramName].numpy() for i in range(2)]
                self.assertTrue(np.array_equal(values[0], values[1]))

    # Helper function to test if parameters of batch norm layers were
    # properly transferred between a network and its successor
    def assertEqualBatchNorm(self, stateDict1, stateDict2, sourceNode,
            destNode, args):

        batchNormVars = ['weight', 'bias', 'running_mean', 'running_var',
                'num_batches_tracked']

        # Check each section for the BN layers that should've been transferred
        for section in range(args['numSections']):

            sourcePrefix = 'sections.%d.%d.1.' % (section, sourceNode)
            destPrefix = 'sections.%d.%d.1.' % (section, destNode)

            for batchNormVar in batchNormVars:

                sourceVarName = sourcePrefix + batchNormVar
                destVarName = sourcePrefix + batchNormVar
                sourceValue = stateDict1[sourceVarName].numpy()
                destValue = stateDict1[destVarName].numpy()

                self.assertTrue(np.array_equal(sourceValue, destValue))

    """
        Tests whether the corresponding layers between a model and the
        resulting model after a growth step have the same weight values.
    """
    def testGrowthStepWeights_Edge_1_Youngest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_Youngest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_1_Oldest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_Oldest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_1_All(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_All(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_1_Youngest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_2_Youngest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_1_Oldest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_2_Oldest(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_1_All(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_2_All(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_All_BN(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        batchNormComparisons = [
            {0: [3, 4], 1: [5, 6]},
            {0: [7, 8, 9, 10], 3: [11, 12], 4: [13, 14], 1: [15, 16, 17, 18],
                5: [19, 20], 6: [21, 22]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons)

    def testGrowthStepWeights_Node_2_All_BN(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False

        batchNormComparisons = [
            {1: [3, 4]},
            {1: [5, 6], 3: [7, 8], 4: [9, 10]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons)

    def compareGrowthWeights(self, args, batchNormComparisons=[]):

        controller = GrowthController(**args)
        stateDicts = []
        nodes = []

        # Grow model and get state dicts
        for i in range(args['growthSteps']):
            oldModel = None if i == 0 else model
            model = controller.step(oldModel=oldModel)
            stateDicts.append(model.state_dict())
            nodes.append(list(model.compGraph.nodes))

        # Compare features and classifier for consecutive steps
        for i in range(args['growthSteps'] - 1):
            self.assertEqualLayers(stateDicts[i], stateDicts[i + 1], nodes[i],
                    args['numSections'])
            self.assertEqualClassifier(stateDicts[i], stateDicts[i + 1])

            # Compare batch norm parameters
            if len(batchNormComparisons) == 0:
                continue

            for sourceBnNode in batchNormComparisons[i].keys():

                destBnNodes = batchNormComparisons[i][sourceBnNode]
                for destBnNode in destBnNodes:
                    self.assertEqualBatchNorm(stateDicts[i], stateDicts[i + 1],
                            sourceBnNode, destBnNode, args)


    """
        Tests whether a model and the resulting model after a growth step
        calculate the same output given the same input.
    """
    def testGrowthStepFunction_Edge_1_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Edge_2_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Edge_1_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Edge_2_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Edge_1_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Edge_2_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Node_1_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Node_2_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Node_1_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Node_2_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Node_1_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Node_2_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        self.compareGrowthFunction(args)

    def compareGrowthFunction(self, args):

        controller = GrowthController(**args)
        outputs = []

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Grow model and run forward passes
        for i in range(args['growthSteps']):
            oldModel = None if i == 0 else model
            model = controller.step(oldModel=oldModel)
            model = model.cuda(0)
            outputs.append(model(testInput).detach().cpu().numpy())

        # Compare outputs
        for i in range(args['growthSteps'] - 1):
            self.assertTrue(np.allclose(outputs[i], outputs[i + 1], atol=1e-6))


    """
        Tests whether the growth controller expands the edges in a computation
        graph correctly.
    """
    def testGrowthStepEdges_Edge_1_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 1), (0, 1), (1, 4), (4, 2), (1, 2)],
                [(0, 5), (5, 3), (0, 3), (3, 6), (6, 1), (3, 1), (0, 1),
                    (1, 7), (7, 4), (1, 4), (4, 8), (8, 2), (4, 2), (1, 2)],
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Edge_2_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 4), (4, 1), (0, 1), (1, 5), (5, 6), (6, 2),
                    (1, 2)],
                [(0, 7), (7, 8), (8, 3), (0, 3), (3, 9), (9, 10), (10, 4),
                    (3, 4), (4, 11), (11, 12), (12, 1), (4, 1), (0, 1),
                    (1, 13), (13, 14), (14, 5), (1, 5), (5, 15), (15, 16),
                    (16, 6), (5, 6), (6, 17), (17, 18), (18, 2), (6, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Edge_1_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 1), (0, 1), (1, 4), (4, 2), (1, 2)],
                [(0, 3), (3, 1), (0, 5), (5, 1), (0, 1), (1, 4), (4, 2),
                    (1, 6), (6, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Edge_2_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 4), (4, 1), (0, 1), (1, 5), (5, 6), (6, 2),
                    (1, 2)],
                [(0, 3), (3, 4), (4, 1), (0, 7), (7, 8), (8, 1), (0, 1),
                    (1, 5), (5, 6), (6, 2), (1, 9), (9, 10), (10, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Edge_1_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 1), (0, 1), (1, 4), (4, 2), (1, 2)],
                [(0, 1), (0, 5), (5, 1), (0, 3), (0, 6), (6, 3), (3, 1),
                            (3, 7), (7, 1), (1, 2), (1, 8), (8, 2), (1, 4), (1, 9),
                            (9, 4), (4, 2), (4, 10), (10, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Edge_2_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 1), (0, 3), (3, 4), (4, 1), (1, 2), (1, 5), (5, 6),
                    (6, 2)],
                [(0, 1), (0, 7), (7, 8), (8, 1), (0, 3), (0, 9), (9, 10),
                    (10, 3), (3, 4), (3, 11), (11, 12), (12, 4), (4, 1),
                    (4, 13), (13, 14), (14, 1), (1, 2), (1, 15), (15, 16),
                    (16, 2), (1, 5), (1, 17), (17, 18), (18, 5), (5, 6),
                    (5, 19), (19, 20), (20, 6), (6, 2), (6, 21), (21, 22),
                    (22, 2)]
        ]


        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Node_1_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (0, 1), (3, 2), (1, 2)],
                [(0, 4), (0, 3), (0, 1), (4, 2), (3, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Node_2_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 4), (0, 1), (4, 2), (1, 2)],
                [(0, 5), (5, 6), (0, 3), (3, 7), (3, 4), (6, 4), (0, 1), (7, 8),
                    (8, 2), (4, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Node_1_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (0, 1), (3, 2), (1, 2)],
                [(0, 4), (0, 3), (0, 1), (4, 2), (3, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Node_2_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 4), (0, 1), (4, 2), (1, 2)],
                [(0, 5), (5, 6), (0, 3), (3, 4), (0, 1), (6, 2), (4, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Node_1_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (0, 1), (3, 2), (1, 2)],
                [(0, 5), (0, 4), (0, 3), (0, 1), (5, 2), (4, 2), (3, 2), (1, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Node_2_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 4), (0, 1), (4, 2), (1, 2)],
                [(0, 5), (5, 6), (0, 7), (7, 8), (0, 3), (3, 9), (3, 4), (8, 4), (0, 1), (9, 10),
                    (6, 2), (10, 2), (4, 2), (1, 2)]
        ]


        self.compareGrowthEdges(args, expectedEdges)

    def compareGrowthEdges(self, args, expectedEdges):

        controller = GrowthController(**args)
        actualEdges = []

        # Initialize model
        for i in range(args['growthSteps']):
            oldModel = None if i == 0 else model
            model = controller.step(oldModel=oldModel)
            model = model.cuda(0)
            actualEdges.append(list(model.compGraph.edges))

        # Compare outputs
        for i in range(3):
            actual = set(actualEdges[i])
            expected = set(expectedEdges[i])
            difference = actual ^ expected
            assert len(difference) == 0

    """
        Tests whether the growth controller calculates the correct growth
        history.
    """
    def testGrowthStepHistory_Edge_1_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Edge_2_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2,
                    10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2,
                    18: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Edge_1_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Edge_2_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2,
                    10: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Edge_1_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
                    10: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Edge_2_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2,
                    10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2,
                    18: 2, 19: 2, 20: 2, 21: 2, 22: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Node_1_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Node_2_Youngest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'youngest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Node_1_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Node_2_Oldest(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'oldest'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Node_1_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 1
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Node_2_All(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
                    10: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def compareGrowthHistory(self, args, expectedHistory):

        controller = GrowthController(**args)
        actualHistory = []

        # Initialize model
        for i in range(args['growthSteps']):
            oldModel = None if i == 0 else model
            model = controller.step(oldModel=oldModel)
            model = model.cuda(0)
            actualHistory.append(dict(controller.growthHistory))

        # Compare outputs
        for i in range(3):
            actual = dict(actualHistory[i])
            expected = dict(expectedHistory[i])
            assert actual == expected

    def testDirac(self):

        # Initialize model
        model = SimpleNet()
        model = model.cuda()

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        tensorInput = testInput.cuda(0)

        # Run forward pass
        output = model(tensorInput).detach().cpu().numpy()
        self.assertTrue(np.allclose(output, testInput))


def getActivations(testInput, model):

    currentActivations = [testInput]

    features = model.features._modules.values()
    for feature in features:
        prevLayer = currentActivations[-1]
        currentActivations.append(feature(prevLayer))

    classifierLayers = model.classifier._modules.values()
    for i, layer in enumerate(classifierLayers):
        prevLayer = currentActivations[-1]
        if i == 0:
            prevLayer = prevLayer.view(prevLayer.size(0), -1)
        currentActivations.append(layer(prevLayer))

    for i in range(len(currentActivations)):
        currentActivations[i] = currentActivations[i].detach().cpu().numpy()

    return list(currentActivations)


def maxDiff(arr1, arr2):
    return np.amax(np.absolute(arr1 - arr2))

if __name__ == '__main__':
    unittest.main()

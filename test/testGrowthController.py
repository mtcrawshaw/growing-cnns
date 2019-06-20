import sys
import unittest
import importlib

import torch
import numpy as np

from testUtils import getTestInput
from toyModel import SimpleNet

sys.path.append('../growing-cnns/architecture')
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


    """
        Tests whether the corresponding layers between a model and the
        resulting model after a growth step have the same weight values.
    """
    def testGrowthStepWeights_Linear(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'linear'

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Skip(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skip'

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_SkipSlim(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skipSlim'

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Branching(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'branching'

        self.compareGrowthWeights(args)

    def compareGrowthWeights(self, args):

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


    """
        Tests whether a model and the resulting model after a growth step
        calculate the same output given the same input.
    """
    def testGrowthStepFunction_Linear(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'linear'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Skip(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skip'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_SkipSlim(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skipSlim'

        self.compareGrowthFunction(args)

    def testGrowthStepFunction_Branching(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'branching'

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
            self.assertTrue(np.allclose(outputs[i], outputs[i + 1], atol=1e-7))


    """
        Tests whether the growth controller expands the edges in a computation
        graph correctly.
    """
    def testGrowthStepEdges_Linear(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'linear'
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 1), (1, 4), (4, 2)],
                [(0, 5), (5, 3), (3, 6), (6, 1), (1, 7), (7, 4), (4, 8), (8, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Skip(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skip'
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 1), (0, 1), (1, 4), (4, 2), (1, 2)],
                [(0, 1), (0, 5), (5, 1), (0, 3), (0, 6), (6, 3), (3, 1),
                            (3, 7), (7, 1), (1, 2), (1, 8), (8, 2), (1, 4), (1, 9),
                            (9, 4), (4, 2), (4, 10), (10, 2)]
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_SkipSlim(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skipSlim'
        expectedEdges = [
                [(0, 1), (1, 2)],
                [(0, 3), (3, 1), (0, 1), (1, 4), (4, 2), (1, 2)],
                [(0, 5), (5, 3), (0, 3), (3, 6), (6, 1), (3, 1), (0, 1),
                    (1, 7), (7, 4), (1, 4), (4, 8), (8, 2), (4, 2), (1, 2)],
        ]

        self.compareGrowthEdges(args, expectedEdges)

    def testGrowthStepEdges_Branching(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'branching'
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
    def testGrowthStepHistory_Linear(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'linear'
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Skip(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skip'
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
                    10: 2},
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_SkipSlim(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'skipSlim'
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
        ]

        self.compareGrowthHistory(args, expectedHistory)

    def testGrowthStepHistory_Branching(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['growthMode'] = 'branching'
        expectedHistory = [
                {0: 0, 1: 0, 2: 0},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2,
                    10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2,
                    18: 2, 19: 2, 20: 2, 21: 2, 22: 2},
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

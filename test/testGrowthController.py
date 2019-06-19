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
    def testGrowthStepWeights(self):

        args = {}
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128

        stateDicts = []
        controller = GrowthController(**args)

        nodes = []

        # Initialize model
        model = controller.step()
        stateDicts.append(model.state_dict())
        nodes.append(list(model.compGraph.nodes))

        # Growth step 1
        model = controller.step(oldModel=model)
        stateDicts.append(model.state_dict())
        nodes.append(list(model.compGraph.nodes))

        # Growth step 2
        model = controller.step(oldModel=model)
        stateDicts.append(model.state_dict())
        nodes.append(list(model.compGraph.nodes))

        # Compare features and classifier for steps 0 and 1
        self.assertEqualLayers(stateDicts[0], stateDicts[1], nodes[0],
                args['maxPools'])
        self.assertEqualClassifier(stateDicts[0], stateDicts[1])

        # Compare features and classifier for steps 1 and 2
        self.assertEqualLayers(stateDicts[1], stateDicts[2], nodes[1],
                args['maxPools'])
        self.assertEqualClassifier(stateDicts[1], stateDicts[2])


    """
        Tests whether a model and the resulting model after a growth step
        calculate the same output given the same input.
    """
    def testGrowthStepFunction(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        controller = GrowthController(**args)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Initialize model
        outputs = []
        model = controller.step()
        model = model.cuda(0)
        outputs.append(model(testInput).detach().cpu().numpy())

        # Growth step 1
        model = controller.step(oldModel=model)
        model = model.cuda(0)
        outputs.append(model(testInput).detach().cpu().numpy())

        # Growth step 2
        model = controller.step(oldModel=model)
        model = model.cuda(0)
        outputs.append(model(testInput).detach().cpu().numpy())

        # Compare outputs
        self.assertTrue(np.allclose(outputs[0], outputs[1], atol=1e-7))
        self.assertTrue(np.allclose(outputs[1], outputs[2], atol=1e-7))


    """
        Tests whether the growth controller expands the edges in a computation
        graph correctly.
    """
    def testGrowthStepEdges(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        controller = GrowthController(**args)

        actualEdges = []
        expectedEdges = []

        # Initialize model
        model = controller.step()
        model = model.cuda(0)
        actualEdges.append(list(model.compGraph.edges))
        expectedEdges.append([(0, 1), (1, 2)])

        # Growth step 1
        model = controller.step(oldModel=model)
        model = model.cuda(0)
        actualEdges.append(list(model.compGraph.edges))
        expectedEdges.append([(0, 1), (1, 3), (3, 2)])

        # Growth step 2
        model = controller.step(oldModel=model)
        model = model.cuda(0)
        actualEdges.append(list(model.compGraph.edges))
        expectedEdges.append([(0, 1), (1, 4), (4, 3), (3, 5), (5, 2)])

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
    def testGrowthStepHistory(self):

        # Create growth controller
        args = {}
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        controller = GrowthController(**args)

        actualHistory = []
        expectedHistory = []

        # Initialize model
        model = controller.step()
        model = model.cuda(0)
        actualHistory.append(dict(controller.growthHistory))
        expectedHistory.append({0: 0, 1: 0, 2: 0})

        # Growth step 1
        model = controller.step(oldModel=model)
        model = model.cuda(0)
        actualHistory.append(dict(controller.growthHistory))
        expectedHistory.append({0: 0, 1: 0, 2: 0, 3: 1})

        # Growth step 2
        model = controller.step(oldModel=model)
        model = model.cuda(0)
        actualHistory.append(dict(controller.growthHistory))
        expectedHistory.append({0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2})

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

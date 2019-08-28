import os
import sys
import unittest
import importlib

import numpy as np

projectRoot = os.path.join(os.path.dirname(__file__), '..', '..')
testRoot = os.path.join(projectRoot, 'test')
sys.path.append(testRoot)
testUtils = importlib.import_module('testUtils')
getTestInput = testUtils.getTestInput
toyModel = importlib.import_module('toyModel')
SimpleNet = toyModel.SimpleNet

class TestGrowthController(unittest.TestCase):

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

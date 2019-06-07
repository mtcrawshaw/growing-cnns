import sys
import unittest
import importlib

import torch
import numpy as np

from toyModel import SimpleNet

sys.path.append('../growing-cnns')
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
    def assertEqualLayers(self, stateDict1, stateDict2, parallelLayers):

        stateDicts = [stateDict1, stateDict2]

        # Check each layer that should've been transferred
        for layerIndices in parallelLayers:

            # Collect layer parameter names
            paramNames = [None, None]
            for i, stateDict in enumerate(stateDicts):

                prefix = 'features.%d' % layerIndices[i]
                paramNames[i] = [key for key in stateDict.keys() if 
                        prefix in key]

                # Get parameter suffixes (whole names differ because of index)
                paramSuffixes = []
                for paramName in paramNames[i]:
                    secondDot = paramName.index('.', len('features.'))
                    paramSuffixes.append(paramName[secondDot + 1:])
                paramNames[i] = list(paramSuffixes)

            # Compare layer parameter names
            for i in range(2):
                paramNames[i] = set(paramNames[i])
            difference = paramNames[0] ^ paramNames[1]
            self.assertTrue(len(difference) == 0)

            # Compare layer parameter values
            paramNames = paramNames[0]
            for paramName in paramNames:

                # Collect values
                values = [None, None]
                for i in range(2):
                    fullName = 'features.%d.%s' % (layerIndices[i], paramName)
                    values[i] = stateDicts[i][fullName].numpy()

                self.assertTrue(np.array_equal(values[0], values[1]))


    """
        Tests whether the corresponding layers between a model and the
        resulting model after a growth step have the same weight values.
    """
    def testGrowthStepWeights(self):

        args = {}
        args['growthSteps'] = 3
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['convPerMaxPool'] = 2
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128

        stateDicts = []
        controller = GrowthController(**args)

        # Initialize model
        model = controller.step()
        stateDicts.append(model.state_dict())

        # Growth step 1
        model = controller.step(oldModel=model)
        stateDicts.append(model.state_dict())

        # Growth step 2
        model = controller.step(oldModel=model)
        stateDicts.append(model.state_dict())

        # Compare features and classifier for steps 0 and 1
        parallelLayers = [(0, 0), (1, 2), (3, 4), (4, 6), (6, 8), (7, 10),
                (9, 12), (10, 14)]
        self.assertEqualLayers(stateDicts[0], stateDicts[1], parallelLayers)
        self.assertEqualClassifier(stateDicts[0], stateDicts[1])

        # Compare features and classifier for steps 1 and 2
        parallelLayers = [(0, 0), (1, 1), (2, 3), (4, 5), (5, 6), (6, 8),
                (8, 10), (9, 11), (10, 13), (12, 15), (13, 16), (14, 18)]
        self.assertEqualLayers(stateDicts[1], stateDicts[2], parallelLayers)
        self.assertEqualClassifier(stateDicts[1], stateDicts[2])



    """
        Tests whether a model and the resulting model after a growth step
        calculate the same output given the same input.
    """
    def testGrowthStepFunction(self):

        # Create growth controller
        args = {}
        args['growthSteps'] = 3
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['convPerMaxPool'] = 2
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        controller = GrowthController(**args)

        # Create test input
        batchSize = 8
        imageHeight = 32
        imageWidth = 32
        imageDepth = 3
        Z = batchSize + imageHeight + imageWidth + imageDepth
        testInput = np.zeros([batchSize, imageDepth, imageHeight, imageWidth])
        for b in range(batchSize):
            for i in range(imageHeight):
                for j in range(imageWidth):
                    for k in range(imageDepth):
                        testInput[b, k, i, j] = float(b + i + j + k)
        testInput = torch.as_tensor(testInput, dtype=torch.float32)
        testInput = testInput.cuda(0)

        outputs = []

        # Initialize model
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
        self.assertTrue(np.allclose(outputs[1], outputs[2]))
        self.assertTrue(np.allclose(outputs[1], outputs[2]))


    def testDirac(self):

        # Initialize model
        model = SimpleNet()
        model = model.cuda()

        # Create test input
        batchSize = 8
        imageHeight = 32
        imageWidth = 32
        imageDepth = 3
        Z = batchSize + imageHeight + imageWidth + imageDepth
        testInput = np.zeros([batchSize, imageDepth, imageHeight, imageWidth])
        for b in range(batchSize):
            for i in range(imageHeight):
                for j in range(imageWidth):
                    for k in range(imageDepth):
                        testInput[b, k, i, j] = float(b + i + j + k) / Z
        tensorInput = torch.as_tensor(testInput, dtype=torch.float32)
        tensorInput = tensorInput.cuda(0)

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

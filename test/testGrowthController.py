import sys
import unittest
import importlib

import numpy as np

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


    def testGrowthStepWeights(self):

        args = {}
        args['growthSteps'] = 3
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['convPerMaxPool'] = 2
        args['numClasses'] = 1000
        args['batchNorm'] = True
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

if __name__ == '__main__':
    unittest.main()

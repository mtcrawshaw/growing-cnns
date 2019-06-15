import sys
import unittest
import importlib

import torch
import numpy as np

sys.path.append('../growing-cnns/architecture')
model = importlib.import_module('model')
CustomConvNet = model.CustomConvNet

class TestModel(unittest.TestCase):

    """
        Tests whether CustomConvNet.forward calculates a forward pass
        correctly.
    """
    def testModelForward(self):

        # Create model
        args = {}
        args['edges'] = [(0, 1), (1, 2)]
        args['convPerSection'] = 3
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        model = CustomConvNet(**args)
        model = model.cuda(0)

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

        model(testInput)

if __name__ == '__main__':
    unittest.main()

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


if __name__ == '__main__':
    unittest.main()

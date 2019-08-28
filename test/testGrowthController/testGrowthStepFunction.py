import os
import sys
import unittest
import importlib

import torch
import numpy as np

projectRoot = os.path.join(os.path.dirname(__file__), '..', '..')
testRoot = os.path.join(projectRoot, 'test')
sys.path.append(testRoot)
testUtils = importlib.import_module('testUtils')
getTestInput = testUtils.getTestInput

packageRoot = os.path.join(projectRoot, 'growingCNNs')
sys.path.append(packageRoot)
growthController = importlib.import_module('growthController')
GrowthController = growthController.GrowthController

class TestGrowthStepFunction(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()

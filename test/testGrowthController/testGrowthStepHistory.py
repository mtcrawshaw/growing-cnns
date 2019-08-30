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

class TestGrowthStepHistory(unittest.TestCase):

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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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
        args['joinType'] = 'uniform'
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

if __name__ == '__main__':
    unittest.main()

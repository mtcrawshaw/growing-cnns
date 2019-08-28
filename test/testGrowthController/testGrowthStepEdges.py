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

class TestGrowthStepEdges(unittest.TestCase):

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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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
        args['joinWeighting'] = 'uniform'
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

if __name__ == '__main__':
    unittest.main()


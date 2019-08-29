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

class TestGrowthStepWeights(unittest.TestCase):

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
    def assertEqualLayers(self, stateDict1, stateDict2, oldNodes, numSections,
            testJoinWeights=False):

        stateDicts = [stateDict1, stateDict2]

        # Check each layer that should've been transferred
        for section in range(numSections):

            # Collect layer parameter names
            paramNames = [None, None]
            for i, stateDict in enumerate(stateDicts):
                for node in oldNodes:

                    paramNames[i] = []
                    prefixes = [
                        'sections.%d.%d' % (section, node),
                        'joinWeights.%d.%d' % (section, node)
                    ]
                    for prefix in prefixes:
                        paramNames[i] += [key for key in stateDict.keys() if 
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

                if 'sections' in paramName:
                    self.assertTrue(np.array_equal(values[0], values[1]))
                elif 'joinWeights' in paramName:
                    n = min(len(values[0]), len(values[1]))
                    if testJoinWeights:
                        self.assertTrue(np.array_equal(values[0][:n], values[1][:n]))

    # Helper function to test if parameters of batch norm layers were
    # properly transferred between a network and its successor
    def assertEqualBatchNorm(self, stateDict1, stateDict2, sourceNode,
            destNode, args):

        batchNormVars = ['weight', 'bias', 'running_mean', 'running_var',
                'num_batches_tracked']

        # Check each section for the BN layers that should've been transferred
        for section in range(args['numSections']):

            sourcePrefix = 'sections.%d.%d.1.' % (section, sourceNode)
            destPrefix = 'sections.%d.%d.1.' % (section, destNode)

            for batchNormVar in batchNormVars:

                sourceVarName = sourcePrefix + batchNormVar
                destVarName = sourcePrefix + batchNormVar
                sourceValue = stateDict1[sourceVarName].numpy()
                destValue = stateDict1[destVarName].numpy()

                self.assertTrue(np.array_equal(sourceValue, destValue))

    """
        Tests whether the corresponding layers between a model and the
        resulting model after a growth step have the same weight values.
    """
    def testGrowthStepWeights_Edge_1_Youngest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_Youngest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_1_Oldest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_Oldest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_1_All_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_All_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_1_Youngest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_2_Youngest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_1_Oldest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_2_Oldest_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_1_All_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Node_2_All_Uniform(self):

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

        self.compareGrowthWeights(args)

    def testGrowthStepWeights_Edge_2_All_BN_Uniform(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        batchNormComparisons = [
            {0: [3, 4], 1: [5, 6]},
            {0: [7, 8, 9, 10], 3: [11, 12], 4: [13, 14], 1: [15, 16, 17, 18],
                5: [19, 20], 6: [21, 22]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons)

    def testGrowthStepWeights_Node_2_All_BN_Uniform(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'

        batchNormComparisons = [
            {1: [3, 4]},
            {1: [5, 6], 3: [7, 8], 4: [9, 10]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons)

    def testGrowthStepWeights_Edge_1_Youngest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_Youngest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_1_Oldest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_Oldest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_1_All_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_All_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_1_Youngest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_2_Youngest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_1_Oldest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_2_Oldest_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_1_All_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_2_All_Softmax(self):

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
        args['joinWeighting'] = 'softmax'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_All_BN_Softmax(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'softmax'

        batchNormComparisons = [
            {0: [3, 4], 1: [5, 6]},
            {0: [7, 8, 9, 10], 3: [11, 12], 4: [13, 14], 1: [15, 16, 17, 18],
                5: [19, 20], 6: [21, 22]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons,
                testJoinWeights=True)

    def testGrowthStepWeights_Node_2_All_BN_Softmax(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'softmax'

        batchNormComparisons = [
            {1: [3, 4]},
            {1: [5, 6], 3: [7, 8], 4: [9, 10]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons,
                testJoinWeights=True)

    def testGrowthStepWeights_Edge_1_Youngest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_Youngest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_1_Oldest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_Oldest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_1_All_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_All_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_1_Youngest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_2_Youngest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_1_Oldest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_2_Oldest_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_1_All_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Node_2_All_Free(self):

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
        args['joinWeighting'] = 'free'

        self.compareGrowthWeights(args, testJoinWeights=True)

    def testGrowthStepWeights_Edge_2_All_BN_Free(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandEdge'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'free'

        batchNormComparisons = [
            {0: [3, 4], 1: [5, 6]},
            {0: [7, 8, 9, 10], 3: [11, 12], 4: [13, 14], 1: [15, 16, 17, 18],
                5: [19, 20], 6: [21, 22]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons,
                testJoinWeights=True)

    def testGrowthStepWeights_Node_2_All_BN_Free(self):

        args = {}
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['initialNumNodes'] = 3
        args['growthSteps'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = True
        args['growthMode'] = 'expandNode'
        args['numConvToAdd'] = 2
        args['itemsToExpand'] = 'all'
        args['copyBatchNorm'] = True
        args['randomWeights'] = False
        args['joinWeighting'] = 'free'

        batchNormComparisons = [
            {1: [3, 4]},
            {1: [5, 6], 3: [7, 8], 4: [9, 10]}
        ]

        self.compareGrowthWeights(args,
                batchNormComparisons=batchNormComparisons,
                testJoinWeights=True)

    def compareGrowthWeights(self, args, batchNormComparisons=[],
            testJoinWeights=False):

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
                    args['numSections'], testJoinWeights)
            self.assertEqualClassifier(stateDicts[i], stateDicts[i + 1])

            # Compare batch norm parameters
            if len(batchNormComparisons) == 0:
                continue

            for sourceBnNode in batchNormComparisons[i].keys():

                destBnNodes = batchNormComparisons[i][sourceBnNode]
                for destBnNode in destBnNodes:
                    self.assertEqualBatchNorm(stateDicts[i], stateDicts[i + 1],
                            sourceBnNode, destBnNode, args)

if __name__ == '__main__':
    unittest.main()

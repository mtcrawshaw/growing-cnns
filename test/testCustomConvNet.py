import sys
import unittest
import importlib
from math import log

import torch
import numpy as np

from testUtils import getTestInput

sys.path.append('../growingCNNs/')
customConvNet = importlib.import_module('customConvNet')
CustomConvNet = customConvNet.CustomConvNet

sys.path.append('../growingCNNs/utils/')
computationGraph = importlib.import_module('computationGraph')
ComputationGraph = computationGraph.ComputationGraph

class TestCustomConvNet(unittest.TestCase):

    """
        Tests whether CustomConvNet.forward calculates a forward pass
        correctly.
    """
    def testCustomConvNetForward_Uniform_1(self):

        # Create model
        args = {}
        args['compGraph'] = ComputationGraph(
                [(0, 1), (1, 2)],
                0,
                2
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Uniform_2(self):

        # Create model
        args = {}
        numNodes = 4
        args['compGraph'] = ComputationGraph(
                [(0, 1), (0, 2), (1, 3), (2, 3)],
                0,
                3
        )
        args['initialChannels'] = 3
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        convParams = [
                (2., 1.),
                (0.5, 3.),
                (1., 2.),
                (3., -1.),
        ]
        stateDict = model.state_dict()
        for section in range(args['numSections']):
            for nodeIndex in range(numNodes):

                # Set convolutional weights
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = convParams[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= 4.5
                            expectedOutput[b, c, x, y] += 8.75

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Uniform_3(self):

        # Create model
        args = {}
        numNodes = 10
        edges = [
            (0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (3, 6),
            (4, 7), (5, 7), (7, 8), (6, 8), (8, 9), (1, 9)
        ]
        args['compGraph'] = ComputationGraph(
                edges,
                0,
                9
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'uniform'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        convParams = [
                (1., 0.),
                (2., 1.),
                (2., -1.),
                (.5, 0.),
                (.5, 1.),
                (1., 1.),
                (2., -1.),
                (.5, 0.),
                (2., -1.),
                (3., 5.)
        ]
        stateDict = model.state_dict()
        for section in range(args['numSections']):
            for nodeIndex in range(numNodes):
            
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = convParams[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= 81. / 16.
                            expectedOutput[b, c, x, y] += 65. / 16.

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Softmax_1(self):

        # Create model
        args = {}
        args['compGraph'] = ComputationGraph(
                [(0, 1), (1, 2)],
                0,
                2
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'softmax'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Softmax_2(self):

        # Create model
        args = {}
        numNodes = 4
        args['compGraph'] = ComputationGraph(
                [(0, 1), (0, 2), (1, 3), (2, 3)],
                0,
                3
        )
        args['initialChannels'] = 3
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'softmax'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        convParams = [
                (2., 1.),
                (0.5, 2.),
                (4., -1.),
                (2., 2.),
        ]
        joinParams = [
                (1.,),
                (2.,),
                (3.,),
                (log(9.), log(1.)),
        ]
        stateDict = model.state_dict()
        for section in range(args['numSections']):
            for nodeIndex in range(numNodes):

                # Set convolutional weights
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = convParams[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

                # Set join weights
                joinKey = 'joinWeights.%d.%d' % (section, nodeIndex)
                joinArr = np.zeros(stateDict[joinKey].shape, dtype=float)
                for i in range(stateDict[joinKey].shape[0]):
                    joinArr[i] = joinParams[nodeIndex][i]
                stateDict[joinKey] = torch.from_numpy(joinArr).float().cuda()

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= 3.4
                            expectedOutput[b, c, x, y] += 7.1

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Softmax_3(self):

        # Create model
        args = {}
        numNodes = 11
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 9), (2, 4), (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 8), (7, 8), (1, 10), (8, 10), (9, 10)
        ]
        args['compGraph'] = ComputationGraph(
                edges,
                0,
                10
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'softmax'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        convParams = [
                (1., 0.),
                (2., 1.),
                (2., -1.),
                (.5, 0.),
                (.5, 1.),
                (1., 1.),
                (2., 1.),
                (.5, 0.),
                (2., -1.),
                (1., 0.),
                (3., 5.)
        ]
        joinParams = [
                (1.,),
                (2.,),
                (-1.,),
                (1.,),
                (-2.,),
                (-1.,),
                (1.,),
                (log(2.), log(3.)),
                (log(2.), log(2.)),
                (1.,),
                (log(3.), log(1.), log(2.))
        ]
        stateDict = model.state_dict()
        for section in range(args['numSections']):
            for nodeIndex in range(numNodes):

                # Set convolutional weights
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = convParams[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

                # Set join weights
                joinKey = 'joinWeights.%d.%d' % (section, nodeIndex)
                joinArr = np.zeros(stateDict[joinKey].shape, dtype=float)
                for i in range(stateDict[joinKey].shape[0]):
                    joinArr[i] = joinParams[nodeIndex][i]
                stateDict[joinKey] = torch.from_numpy(joinArr).float().cuda()

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= 10.75
                            expectedOutput[b, c, x, y] += 6.8

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Free_1(self):

        # Create model
        args = {}
        args['compGraph'] = ComputationGraph(
                [(0, 1), (1, 2)],
                0,
                2
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'free'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Free_2(self):

        # Create model
        args = {}
        numNodes = 4
        args['compGraph'] = ComputationGraph(
                [(0, 1), (0, 2), (1, 3), (2, 3)],
                0,
                3
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'free'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        convParams = [
                (2., -1.),
                (0.5, 1.),
                (1., 0.),
                (.5, 2.),
        ]
        joinParams = [
                (2.,),
                (0.5,),
                (1.,),
                (2., 3.),
        ]
        stateDict = model.state_dict()
        for section in range(args['numSections']):
            for nodeIndex in range(numNodes):

                # Set convolutional weights
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = convParams[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

                # Set join weights
                joinKey = 'joinWeights.%d.%d' % (section, nodeIndex)
                joinArr = np.zeros(stateDict[joinKey].shape, dtype=float)
                for i in range(stateDict[joinKey].shape[0]):
                    joinArr[i] = joinParams[nodeIndex][i]
                stateDict[joinKey] = torch.from_numpy(joinArr).float().cuda()

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= 7.
                            expectedOutput[b, c, x, y] += 1.25

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testCustomConvNetForward_Free_3(self):

        # Create model
        args = {}
        numNodes = 11
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 9), (2, 4), (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 8), (7, 8), (1, 10), (8, 10), (9, 10)
        ]
        args['compGraph'] = ComputationGraph(
                edges,
                0,
                10
        )
        args['initialChannels'] = 8
        args['numSections'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['randomWeights'] = False
        args['joinWeighting'] = 'free'
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        convParams = [
                (1., 0.),
                (2., 1.),
                (2., -1.),
                (.5, 0.),
                (.5, 1.),
                (1., 1.),
                (2., -1.),
                (.5, 0.),
                (2., 5.),
                (1., 0.),
                (3., 5.)
        ]
        joinParams = [
                (1.,),
                (2.,),
                (-1.,),
                (1.,),
                (-2.,),
                (-1.,),
                (1.,),
                (2., -2.),
                (1., -1.),
                (1.,),
                (3., 2., 1.)
        ]
        stateDict = model.state_dict()
        for section in range(args['numSections']):
            for nodeIndex in range(numNodes):
            
                # Set convolutional weights
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = convParams[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

                # Set join weights
                joinKey = 'joinWeights.%d.%d' % (section, nodeIndex)
                joinArr = np.zeros(stateDict[joinKey].shape, dtype=float)
                for i in range(stateDict[joinKey].shape[0]):
                    joinArr[i] = joinParams[nodeIndex][i]
                stateDict[joinKey] = torch.from_numpy(joinArr).float().cuda()

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['numSections'])
        outputShape[2] = outputShape[2] // (2 ** args['numSections'])
        outputShape[3] = outputShape[3] // (2 ** args['numSections'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= 33.
                            expectedOutput[b, c, x, y] += 20.

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))

if __name__ == '__main__':
    unittest.main()

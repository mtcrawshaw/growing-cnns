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
        Helper function which generates an example input tensor
    """
    def getTestInput(self, inputShape):

        # Create test input
        batchSize = inputShape[0]
        imageDepth = inputShape[1]
        imageWidth = inputShape[2]
        imageHeight = inputShape[3]

        Z = batchSize + imageHeight + imageWidth + imageDepth
        testInput = np.zeros([batchSize, imageDepth, imageHeight, imageWidth])
        for b in range(batchSize):
            for i in range(imageHeight):
                for j in range(imageWidth):
                    for k in range(imageDepth):
                        testInput[b, k, i, j] = float(b + i + j + k)
        testInput = torch.as_tensor(testInput, dtype=torch.float32)
        return testInput


    """
        Tests whether CustomConvNet.forward calculates a forward pass
        correctly.
    """
    def testModelForward_1(self):

        # Create model
        args = {}
        args['edges'] = [(0, 1), (1, 2)]
        args['convPerSection'] = 3
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['randomWeights'] = False
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = self.getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['maxPools'])
        outputShape[2] = outputShape[2] // (2 ** args['maxPools'])
        outputShape[3] = outputShape[3] // (2 ** args['maxPools'])
        expectedOutput = np.zeros(outputShape)
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['maxPools']) - 1
                        maxY = (y + 1) * (2 ** args['maxPools']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testModelForward_2(self):

        # Create model
        args = {}
        args['edges'] = [(0, 1), (1, 2), (2, 3)]
        args['convPerSection'] = 4
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['randomWeights'] = False
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        stateDict = model.state_dict()
        for key in stateDict.keys():
            if 'sections' in key and 'weight' in key:
                stateDict[key] *= 2
        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = self.getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['maxPools'])
        outputShape[2] = outputShape[2] // (2 ** args['maxPools'])
        outputShape[3] = outputShape[3] // (2 ** args['maxPools'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = args['convPerSection'] * args['maxPools']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['maxPools']) - 1
                        maxY = (y + 1) * (2 ** args['maxPools']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY
                        expectedOutput[b, c, x, y] *= 2 ** numTotalConv

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))


    def testModelForward_3(self):

        # Create model
        args = {}
        args['edges'] = [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (3, 6),
                (4, 7), (5, 7), (7, 8), (6, 8), (8, 9), (1, 9)]
        args['convPerSection'] = 10
        args['initialChannels'] = 8
        args['maxPools'] = 3
        args['numClasses'] = 1000
        args['batchNorm'] = False
        args['classifierHiddenSize'] = 128
        args['randomWeights'] = False
        model = CustomConvNet(**args)
        model = model.cuda(0)

        # Set model parameters
        params = [
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
        for section in range(args['maxPools']):
            for nodeIndex in range(args['convPerSection']):
            
                keyPrefix = 'sections.%d.%d.0.' % (section, nodeIndex)
                weightKey = keyPrefix + 'weight'
                biasKey = keyPrefix + 'bias'

                weight, bias = params[nodeIndex]
                biasArr = np.zeros(stateDict[biasKey].shape, dtype=float)
                for i in range(3):
                    biasArr[i] = bias

                stateDict[weightKey] *= weight
                biasTensor = torch.from_numpy(biasArr).float().cuda()
                stateDict[biasKey] += biasTensor

        model.load_state_dict(stateDict)

        # Create test input
        inputShape = [8, 3, 32, 32]
        testInput = self.getTestInput(inputShape)
        testInput = testInput.cuda(0)

        # Run forward pass
        output = model.convForward(testInput)

        # Build expected output
        outputShape = list(inputShape)
        outputShape[1] = args['initialChannels'] * (2 ** args['maxPools'])
        outputShape[2] = outputShape[2] // (2 ** args['maxPools'])
        outputShape[3] = outputShape[3] // (2 ** args['maxPools'])
        expectedOutput = np.zeros(outputShape)
        numTotalConv = args['convPerSection'] * args['maxPools']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['maxPools']) - 1
                        maxY = (y + 1) * (2 ** args['maxPools']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['maxPools']):
                            expectedOutput[b, c, x, y] *= 81. / 16.
                            expectedOutput[b, c, x, y] += 65. / 16.

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))

if __name__ == '__main__':
    unittest.main()

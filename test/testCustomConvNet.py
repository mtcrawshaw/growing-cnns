import sys
import unittest
import importlib
from math import log

import torch
import numpy as np

from testUtils import getTestInput, setWeights

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

        convParams = {}
        joinParams = {}
        expectedAffine = (1., 0.)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Uniform_2(self):

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

        convParams = {
                0: (2., 1.),
                1: (0.5, 3.),
                2: (1., 2.),
                3: (3., -1.),
        }
        joinParams = {}
        expectedAffine = (4.5, 8.75)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Uniform_3(self):

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

        convParams = {
                0: (1., 0.),
                1: (2., 1.),
                2: (2., -1.),
                3: (.5, 0.),
                4: (.5, 1.),
                5: (1., 1.),
                6: (2., -1.),
                7: (.5, 0.),
                8: (2., -1.),
                9: (3., 5.)
        }
        joinParams = {}
        expectedAffine = (81. / 16., 65. / 16.)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Softmax_1(self):

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

        convParams = {}
        joinParams = {}
        expectedAffine = (1., 0.)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Softmax_2(self):

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

        convParams = {
                0: (2., 1.),
                1: (0.5, 2.),
                2: (4., -1.),
                3: (2., 2.),
        }
        joinParams = {
                0: (1.,),
                1: (2.,),
                2: (3.,),
                3: (log(9.), log(1.)),
        }
        expectedAffine = (3.4, 7.1)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Softmax_3(self):

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

        convParams = {
                0: (1., 0.),
                1: (2., 1.),
                2: (2., 2.),
                3: (.5, 0.),
                4: (.5, 1.),
                5: (1., 1.),
                6: (2., 1.),
                7: (.5, 0.),
                8: (2., -1.),
                9: (1., 0.),
                10: (3., 5.),
        }
        joinParams = {
                0: (1.,),
                1: (2.,),
                2: (-1.,),
                3: (1.,),
                4: (-2.,),
                5: (-1.,),
                6: (1.,),
                7: (log(2.), log(3.)),
                8: (log(2.), log(2.)),
                9: (1.,),
                10: (log(3.), log(1.), log(2.))
        }
        expectedAffine = (4.675, 6.85)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Free_1(self):

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

        convParams = {}
        joinParams = {}
        expectedAffine = (1., 0.)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Free_2(self):

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

        convParams = {
                0: (2., -1.),
                1: (0.5, 1.),
                2: (1., 0.),
                3: (.5, 2.),
        }
        joinParams = {
                0: (2.,),
                1: (0.5,),
                2: (1.,),
                3: (2., 3.),
        }
        expectedAffine = (7., 1.25)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def testCustomConvNetForward_Free_3(self):

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

        convParams = {
                0: (1., 0.),
                1: (2., 1.),
                2: (-2., 1.),
                3: (.5, 0.),
                4: (-.5, 2.),
                5: (-2., 1.),
                6: (2., 0.),
                7: (.5, 0.),
                8: (2., 5.),
                9: (1., 0.),
                10: (3., 5.)
        }
        joinParams = {
                0: (1.,),
                1: (2.,),
                2: (-1.,),
                3: (1.,),
                4: (-2.,),
                5: (-1.,),
                6: (1.,),
                7: (2., -1.),
                8: (-1., 1.),
                9: (1.,),
                10: (3., 2., 1.)
        }
        expectedAffine = (45., 74.)

        self.compareForward(args, convParams, joinParams, expectedAffine)

    def compareForward(self, args, convParams, joinParams, expectedAffine):

        model = CustomConvNet(**args)
        model = model.cuda(0)

        model = setWeights(
                model,
                convParams,
                joinParams
        )

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
        numTotalConv = args['compGraph'].numNodes * args['numSections']
        for b in range(outputShape[0]):
            for c in range(3):
                for x in range(outputShape[2]):
                    for y in range(outputShape[3]):
                        maxX = (x + 1) * (2 ** args['numSections']) - 1
                        maxY = (y + 1) * (2 ** args['numSections']) - 1
                        expectedOutput[b, c, x, y] = b + c + maxX + maxY

                        for i in range(args['numSections']):
                            expectedOutput[b, c, x, y] *= expectedAffine[0]
                            expectedOutput[b, c, x, y] += expectedAffine[1]

        # Test output
        outputValue = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(expectedOutput, outputValue))

if __name__ == '__main__':
    unittest.main()

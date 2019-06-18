import torch
import numpy as np

def getTestInput(inputShape):

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


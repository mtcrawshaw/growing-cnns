import torch
import numpy as np

"""
    Creates a tensor of shape inputShape to be used as a dummy input.
"""
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

def setWeights(numSections, stateDict, convParams, joinParams=None):

    for section in range(numSections):
        for nodeIndex in convParams:

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

        if joinParams is None:
            continue
        for nodeIndex in joinParams:

            # Set join weights
            if joinParams is not None:
                joinKey = 'joinWeights.%d.%d' % (section, nodeIndex)
                joinArr = np.zeros(stateDict[joinKey].shape, dtype=float)
                for i in range(stateDict[joinKey].shape[0]):
                    joinArr[i] = joinParams[nodeIndex][i]
                stateDict[joinKey] = torch.from_numpy(joinArr).float().cuda()

    return stateDict


def getActivations(testInput, model):

    currentActivations = [testInput]

    features = model.features._modules.values()
    for feature in features:
        prevLayer = currentActivations[-1]
        currentActivations.append(feature(prevLayer))

    classifierLayers = model.classifier._modules.values()
    for i, layer in enumerate(classifierLayers):
        prevLayer = currentActivations[-1]
        if i == 0:
            prevLayer = prevLayer.view(prevLayer.size(0), -1)
        currentActivations.append(layer(prevLayer))

    for i in range(len(currentActivations)):
        currentActivations[i] = currentActivations[i].detach().cpu().numpy()

    return list(currentActivations)


def maxDiff(arr1, arr2):
    return np.amax(np.absolute(arr1 - arr2))


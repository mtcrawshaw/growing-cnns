import math

from model import CustomConvNet

"""
    Growth Controller
"""

class GrowthController():

    def __init__(self, initialChannels=64, maxPools=4, convPerMaxPool=3,
            growthSteps=3, numClasses=1000, batchNorm=True,
            classifierHiddenSize=2048):
        
        self.numClasses = numClasses
        self.batchNorm = batchNorm
        self.growthSteps = growthSteps
        self.currentStep = -1

        self.initialChannels = initialChannels
        self.maxPools = maxPools
        self.convPerMaxPool = convPerMaxPool
        self.classifierHiddenSize = classifierHiddenSize

        # Define growth history, a list of the growth steps each layer was
        # inserted during. So on step 0, this list will be all zeroes and with
        # length equal to the number of layers on iteration 0. On step 1, this
        # list will contain mostly zeroes, with ones inserted at the positions
        # where new layers were inserted during the growth step, etc.
        self.growthHistory = []

    def step(self, oldModel=None):
        """
        Produces the next step in the growing model from the previous state dictionary
        Args:
            oldModel (CustomConvNet): Model to be grown, that was also returned from step
        """
        assert self.currentStep < self.growthSteps

        # Initially creating model
        if self.currentStep == -1:
            self.currentStep = 0
            newModel = CustomConvNet(self.initialChannels, self.maxPools,
                    self.convPerMaxPool, numClasses=self.numClasses,
                    batchNorm=self.batchNorm,
                    classifierHiddenSize=self.classifierHiddenSize) 
            numLayers = self.maxPools * (self.convPerMaxPool + 1)
            self.growthHistory = [0 for i in range(numLayers)]
            return newModel

        # Create new model
        self.currentStep += 1
        self.convPerMaxPool += 1
        newModel = CustomConvNet(self.initialChannels, self.maxPools,
                self.convPerMaxPool, numClasses=self.numClasses,
                batchNorm=self.batchNorm,
                classifierHiddenSize=self.classifierHiddenSize)

        # Transfer weights from old model to new model
        for i in range(self.maxPools):

            """
            Here we transfer weights into all layers of the new model between
            the ith and (i + 1)th max pooling layer, other than the second to
            last layer. The second to last layer will be initialized to
            calculate the identity function, and it serves as the newly grown
            layer. The reason that the second to last layer is the new one,
            as opposed to the last layer, is because the last layer doubles
            the number of channels (before the width and height are halved by
            the max pool) so a new layer cannot be placed after the last layer.
            """
            newLayerPos = self.convPerMaxPool - 2
            numLayersTransferred = 0

            for j in range(self.convPerMaxPool):

                if j == newLayerPos:
                    self.growthHistory.insert(self.convPerMaxPool * i + j,
                            self.currentStep)
                    continue

                # Grab state dictionary from old layer
                oldLayerIndex = self.convPerMaxPool * i + \
                        numLayersTransferred
                oldLayer = oldModel.features.__getitem__(oldLayerIndex)
                oldStateDict = oldLayer.state_dict()

                # Load old state dictionary into new layer
                newLayerIndex = (self.convPerMaxPool + 1) * i + j
                newLayer = newModel.features.__getitem__(newLayerIndex)
                newLayer.load_state_dict(oldStateDict)
                numLayersTransferred += 1

        # Transfer classifier weights
        newModel.classifier.load_state_dict(oldModel.classifier.state_dict())

        return newModel

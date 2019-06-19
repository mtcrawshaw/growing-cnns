import os
import shutil

import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""
    Sets the learning rate to the initial LR decayed by lrDecayRatio every
    lrDecayStep epochs
"""
def adjustLearningRate(optimizer, lrDecayRatio):
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] *= lrDecayRatio

"""
    Returns optimizer parameters for the beginning of each growth step. The
    learning rate for a given layer decreases by a factor of lrGrowthDecay for
    each growth step that has passed since that layer was introduced to the
    network.
"""
def getInitialOptimizerParams(model, growthHistory, initialLR,
        lrGrowthDecay, growthStep):
    optimizerParams = []

    for section in range(model.numSections):
        for i, layerStep in growthHistory.items():
            lr = initialLR * (lrGrowthDecay ** (growthStep - layerStep))
            optimizerParams.append({
                'params': model.sections[section][i].parameters(),
                'lr': lr})

    classifierLR = initialLR * (lrGrowthDecay ** growthStep)
    optimizerParams.append({'params': model.classifier.parameters(),
        'lr': classifierLR})

    return optimizerParams

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batchSize = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correctK = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correctK.mul_(100.0 / batchSize))
        return res

def saveCheckpoint(experimentDir, experimentName, state, isBest):
    filename = os.path.join(experimentDir, experimentName)
    filename += '.pth'
    bestFilename = os.path.join(experimentDir, experimentName + '_best')
    bestFilename += '.pth'
    torch.save(state, filename)
    if isBest:
        shutil.copyfile(filename, bestFilename)



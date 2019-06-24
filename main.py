import argparse
import os
import random
import time
import warnings
import sys
import json
import shutil
import importlib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from growingCNNs.customConvNet import CustomConvNet
from growingCNNs.growthController import GrowthController
from growingCNNs.graphOperations import getInitialCompGraph, growCompGraph
from growingCNNs.utils.computationGraph import ComputationGraph
import growingCNNs.utils.utils as utils
import plotting.createPlots as createPlots

# Import small dataset
#sys.path.append('/data')
#cifarSmall = importlib.import_module('cifarSmall')
#CIFARSmall = cifarSmall.CIFARSmall
from data.cifarSmall import CIFARSmall

experimentDir = None

def main(args):

    # Load settings file
    with open(args.settingsFile, 'r') as settingsFile:
        settings = json.load(settingsFile)
    experimentType = 'growing' if settings['growing'] else 'static'
    
    # Create experiments directory if it doesn't already exist
    experimentsDir = os.path.join(os.path.dirname(__file__), 'experiments')
    if not args.quiet and not os.path.isdir(experimentsDir):
        os.makedirs(experimentsDir)

    # Create experiment directory to store results and model, if args.quiet
    # is false and we are training, not evaluating a model.
    global experimentDir
    experimentDir = os.path.join(experimentsDir, args.name)
    if not args.quiet and args.modelPath is None:
        if os.path.isdir(experimentDir):
            print("Experiment with name '%s' already exists!" % args.name)
            exit()
        else:
            os.makedirs(experimentDir)
            permanentSettingsFile = os.path.join(experimentDir, args.name + '_settings.json')
            shutil.copyfile(args.settingsFile, permanentSettingsFile)

    # Set seed and other cuda settings
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    numClasses = 10 # Temporary
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading
    IMAGENET_MEANS = (0.485, 0.456, 0.406)
    IMAGENET_STDS = (0.229, 0.224, 0.225)

    # Data transformations
    transformTrain = transforms.Compose([
        #transforms.Resize(256),      # Note: These commented transformations should be added for imagenet when the time comes.
        #transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS),
    ])
    transformVal = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS),
    ])

    # Load dataset
    if not args.small:
        trainDataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transformTrain)
        valDataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transformVal)
    else:
        trainDataset = CIFARSmall(root='data', train=True, transform=transformTrain)
        valDataset = CIFARSmall(root='data', train=False, transform=transformVal)

    # Run training/evaluation
    if experimentType == "growing":
        results = runGrowing(numClasses, args, settings, criterion,
                trainDataset, valDataset)
    else:
        results = runStatic(numClasses, args, settings, criterion,
                trainDataset, valDataset)

    # Write out results to log if this is a training session and not quiet mode
    if args.modelPath is None and not args.quiet:
        logPath = os.path.join(experimentDir, '%s.log' % args.name)
        with open(logPath, 'w') as logFile:
            json.dump(results, logFile, indent=4)

        createPlots.main(experimentName=args.name, phase='train')
        createPlots.main(experimentName=args.name, phase='validate')

def runStatic(numClasses, args, settings, criterion, trainDataset, valDataset):

    # Create model. Here we grow a model to full size without any training
    # until the model is full-grown.
    growthController = GrowthController(
            settings['initialChannels'],
            settings['numSections'],
            settings['initialNumNodes'],
            settings['growthSteps'],
            numClasses,
            settings['batchNorm'],
            settings['growthMode'],
            randomWeights=True
    )
    model = None
    for growthStep in range(settings['growthSteps']):
        oldModel = None if growthStep == 0 else model
        model = growthController.step(oldModel=model)
    model = model.cuda(args.gpu)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), settings['initialLearningRate'],
                               momentum=settings['momentum'],
                               weight_decay=settings['weightDecay'])

    # Create train and validation loaders from dataset
    trainSampler = None
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=settings['batchSize'], shuffle=(trainSampler is None),
        num_workers=args.workers, pin_memory=True, sampler=trainSampler)
    valLoader = torch.utils.data.DataLoader(
        valDataset, batch_size=settings['batchSize'], shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Run evaluation only
    if args.modelPath is not None:
        validate(valLoader, model, criterion, 0, args, [])
        return

    # Results object to write out
    results = {}
    trainResults = []
    validateResults = []

    # Create stopping criteria
    stoppingSettings = settings['stoppingCriteria']
    stoppingType = stoppingSettings['type']
    assert stoppingType in ['fixedEpochs', 'lossConvergence']
    stoppingCriteria = None
    if stoppingType == 'fixedEpochs':
        numEpochs = stoppingSettings['numEpochs']
        stoppingCriteria = "epoch >= numEpochs"
    elif stoppingType == 'lossConvergence':
        testWindow = stoppingSettings['testWindow']
        maxEpochs = stoppingSettings['maxEpochs']
        stoppingCriteria = "lastMinimum >= testWindow or epoch >= maxEpochs"

    # Training loop
    bestAcc1 = 0
    epoch = 0
    lastMinimum = 0

    while not eval(stoppingCriteria):
        if epoch > 0 and epoch % settings['lrDecayStep'] == 0:
            utils.adjustLearningRate(optimizer, settings['lrDecayRatio'])

        # Train for one epoch
        train(trainLoader, model, criterion, optimizer, epoch, args, trainResults)

        # Evaluate on validation set
        acc1 = validate(valLoader, model, criterion, epoch, args, validateResults)

        # Remember best acc@1 and save checkpoint
        isBest = acc1 > bestAcc1
        bestAcc1 = max(acc1, bestAcc1)
        lastMinimum = 0 if isBest else lastMinimum + 1
        if isBest:
            print('   New best test accuracy reached!')

        if not args.quiet:
            utils.saveCheckpoint(experimentDir, args.name, {
                'epoch': epoch + 1,
                'stateDict': model.state_dict(),
                'bestAcc1': bestAcc1,
                'optimizer': optimizer.state_dict(),
            }, isBest)

        epoch += 1

    results['trainResults'] = list(trainResults)
    results['validateResults'] = list(validateResults)
    return results

def runGrowing(numClasses, args, settings, criterion, trainDataset,
        valDataset):
    
    # Create growth controller
    growthController = GrowthController(
            settings['initialChannels'],
            settings['numSections'],
            settings['initialNumNodes'],
            settings['growthSteps'],
            numClasses,
            settings['batchNorm'],
            settings['growthMode']
    )
    totalEpoch = 0

    # Only evaluate model, no training
    if args.modelPath is not None:

        # Load model checkpoint
        checkpoint = torch.load(args.modelPath)

        # Grow model to correct size
        totalSteps = checkpoint['growthStep']
        for growthStep in range(totalSteps + 1):
            if growthStep == 0:
                model = growthController.step()
            else:
                model = growthController.step(oldModel=model)

            model = model.cuda(args.gpu)

        # Load weights into model
        model.load_state_dict(checkpoint['stateDict'])

        # Create validation loader from dataset
        if isinstance(settings['batchSize'], list):
            batchSize = settings['batchSize'][totalSteps]
        else:
            batchSize = settings['batchSize']
        valLoader = torch.utils.data.DataLoader(
            valDataset, batch_size=batchSize, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        validate(valLoader, model, criterion, 0, args, [],
                growthStep=totalSteps)
        return

    # Results object to write out
    results = {}
    trainResults = []
    validateResults = []

    # Create stopping criteria
    stoppingSettings = settings['stoppingCriteria']
    stoppingType = stoppingSettings['type']
    assert stoppingType in ['fixedEpochs', 'lossConvergence']
    stoppingCriteria = None
    if stoppingType == 'fixedEpochs':
        numEpochs = stoppingSettings['numEpochs']
        stoppingCriteria = "epoch >= numEpochs"
    elif stoppingType == 'lossConvergence':
        testWindow = stoppingSettings['testWindow']
        maxEpochs = stoppingSettings['maxEpochs']
        stoppingCriteria = "lastMinimum >= testWindow or epoch >= maxEpochs"

    # Outer training loop
    bestAcc1 = 0
    for i in range(growthController.growthSteps):

        # Create train and validation loader from dataset
        if isinstance(settings['batchSize'], list):
            batchSize = settings['batchSize'][i]
        else:
            batchSize = settings['batchSize']
        trainSampler=None
        trainLoader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize,
            shuffle=(trainSampler is None), num_workers=args.workers,
            pin_memory=True, sampler=trainSampler)
        valLoader = torch.utils.data.DataLoader(
            valDataset, batch_size=batchSize, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        # Create model and optimizer
        print("\n=> creating growth iteration %d for model" % i)
        if i == 0:
            model = growthController.step()
        else:
            model = growthController.step(oldModel=model)

        model = model.cuda(args.gpu)

        optimizerParams = utils.getInitialOptimizerParams(
                model,
                growthController.growthHistory,
                settings['initialLearningRate'],
                settings['lrDecayGrowthRatio'], 
                i
        )
        optimizer = torch.optim.SGD(
                optimizerParams,
                momentum=settings['momentum'],
                weight_decay=settings['weightDecay']
        )

        # Inner training loop
        epoch = 0
        lastMinimum = 0
        while not eval(stoppingCriteria):
            if epoch > 0 and epoch % settings['lrDecayEpochStep'] == 0:
                utils.adjustLearningRate(optimizer,
                        settings['lrDecayEpochRatio'])

            # train for one epoch
            train(trainLoader, model, criterion, optimizer, epoch, args, trainResults, growthStep=i)

            # evaluate on validation set
            acc1 = validate(valLoader, model, criterion, epoch, args, validateResults, growthStep=i)

            # remember best acc@1 and save checkpoint
            isBest = acc1 > bestAcc1
            bestAcc1 = max(acc1, bestAcc1)
            lastMinimum = 0 if isBest else lastMinimum + 1
            if isBest:
                print('   New best test accuracy reached!')

            if not args.quiet:
                utils.saveCheckpoint(experimentDir, args.name, {
                    'growthStep': i,
                    'epoch': epoch + 1,
                    'stateDict': model.state_dict(),
                    'bestAcc1': bestAcc1,
                    'optimizer' : optimizer.state_dict(),
                }, isBest)

            epoch += 1
            totalEpoch += 1

    results['trainResults'] = list(trainResults)
    results['validateResults'] = list(validateResults)
    return results

def train(trainLoader, model, criterion, optimizer, epoch, args, trainResults, growthStep=None):
    batchTime = utils.AverageMeter()
    dataTime = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainLoader):
        # measure data loading time
        dataTime.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batchTime.update(time.time() - end)
        end = time.time()

        if i % args.printFreq == 0:
            growthMsg = 0 if growthStep is None else growthStep
            print('Epoch: [{0},{1}][{2}/{3}]\t'
                  'Time {batchTime.val:.3f} ({batchTime.avg:.3f})\t'
                  'Data {dataTime.val:.3f} ({dataTime.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   growthMsg, epoch, i, len(trainLoader), batchTime=batchTime,
                   dataTime=dataTime, loss=losses, top1=top1, top5=top5))
            currentResult = {'epoch': epoch, 'iteration': i,
                              'time': batchTime.val, 'loss': losses.val, 'top1': top1.val.item(),
                              'top5': top5.val.item()}
            if growthStep is not None:
                currentResult['growthStep'] = growthStep
            trainResults.append(dict(currentResult))


def validate(valLoader, model, criterion, epoch, args, validateResults, growthStep=None):
    batchTime = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(valLoader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batchTime.update(time.time() - end)
            end = time.time()

            if i % args.printFreq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batchTime.val:.3f} ({batchTime.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(valLoader), batchTime=batchTime, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        currentResult = {'epoch': epoch, 'loss': losses.avg,
                                 'top1': top1.avg.item(), 'top5': top5.avg.item()} 
        if growthStep is not None:
            currentResult['growthStep'] = growthStep
        validateResults.append(dict(currentResult))

    return top1.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Growing CNNs with PyTorch')
    parser.add_argument('name', type=str, help='name of experiment')
    parser.add_argument('settingsFile', type=str, help='name of settings file '
                        'containing hyperparameter and training settings. Example '
                        'settings file is exampleGrowingSettings.json')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--printFreq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--evaluate', dest='modelPath', type=str, default=None,
                        help='evaluate model with path modelPath on validation set')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--quiet', dest='quiet', default=False,
                        action='store_true', help='whether or not to save a' +
                        'results log and copy of settings file.')
    parser.add_argument('--small', dest='small', default=False,
                        action='store_true', help='whether or not to use '
                        ' small version of dataset')
    
    args = parser.parse_args()
    
    main(args)

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

def adjust_learning_rate(optimizer, lr_decay_ratio):
    """Sets the learning rate to the initial LR decayed by lr_decay_ratio every
    lr_decay_step epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_ratio

def get_initial_optimizer_params(model, growth_history, initial_lr,
        lr_growth_decay, growth_step):
    optimizer_params = []

    for i, layer_step in enumerate(growth_history):
        lr = initial_lr * (lr_growth_decay ** (growth_step - layer_step))
        optimizer_params.append({
            'params': model.features.__getitem__(i).parameters(),
            'lr': lr})

    classifier_lr = initial_lr * (lr_growth_decay ** growth_step)
    optimizer_params.append({'params': model.classifier.parameters(),
        'lr': classifier_lr})

    return optimizer_params

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(experimentDir, experimentName, state, is_best):
    filename = os.path.join(experimentDir, experimentName)
    filename += '.pth'
    best_filename = os.path.join(experimentDir, experimentName + '_best')
    best_filename += '.pth'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)



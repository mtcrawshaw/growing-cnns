from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import shutil
import glob

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data

class CIFARSmall(data.Dataset):
    """`CIFARSmall Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    original_base_folder = 'cifar-10-batches-py'
    base_folder = 'cifar-small'
    train_list = ['data_batch_1']
    test_list = ['test_batch']
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):

        super(CIFARSmall, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list

        self.data = []
        self.targets = []

        # create cifar-small if it doesn't already exist
        smallDir = os.path.join(self.root, self.base_folder)
        if not os.path.isdir(smallDir):
            src = os.path.join(self.root, self.original_base_folder)
            shutil.copytree(src, smallDir)

            filesToKeep = self.train_list + self.test_list
            filesToKeep += [self.meta['filename']]
            dataFiles = glob.glob('%s/*' % smallDir)
            for dataFile in dataFiles:
                if os.path.basename(dataFile) not in filesToKeep:
                    os.remove(dataFile)

        # now load the picked numpy arrays
        for file_name in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

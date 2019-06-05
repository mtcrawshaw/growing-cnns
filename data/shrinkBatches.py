import pickle
import numpy as np
import random

batchNames = ['cifar-small/data_batch_1', 'cifar-small/test_batch']

newSize = 100

for batchName in batchNames:

    # Load in batch
    with open(batchName, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    currentSize = len(batch[b'labels'])

    # Choose which elements to keep
    toKeep = random.sample(range(currentSize), k=newSize)

    # Create new batch and copy chosen elements
    newBatch = {}
    newBatch['batch_label'] = batch[b'batch_label']
    newBatch['labels'] = [batch[b'labels'][i] for i in toKeep]
    newBatch['filenames'] = [batch[b'filenames'][i] for i in toKeep]

    # Copy data
    dataShape = batch[b'data'].shape
    newShape = list(dataShape)
    newShape[0] = newSize
    newShape = tuple(newShape)
    newData = np.zeros(newShape)
    for i, chosenIndex in enumerate(toKeep):
        newData[i] = batch[b'data'][chosenIndex]
    newBatch['data'] = newData

    # Write out new batch
    newBatchName = '%s_new' % batchName
    with open(newBatchName, 'wb') as f:
        pickle.dump(newBatch, f)

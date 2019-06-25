import json

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms

#DATA PREPROCESSING

def read_log(filename):

    # Read and parse log into data frame
    with open(filename, encoding='utf-8') as resultsFile:
        results = json.load(resultsFile)

    splits = ['train', 'validate']
    performanceMetrics = ['loss', 'top1']

    # This is kind of hacky, but it's the quickest way to access
    # the print frequency of training without passing it here
    printFrequency = results['trainResults'][1]['iteration']

    # Build dictionary of dfs for each split/metric pair
    dfs = {}
    for split in splits:
        splitKey = '%sResults' % split
        numIterations = len(results[splitKey])

        for metric in performanceMetrics:
            
            metricList = []
            yLabel = '%s_%s' % (split, metric)
            for i in range(numIterations):
                row = []
                row.append(results[splitKey][i][metric])
                row.append(i * printFrequency)

                metricList.append(list(row))

            dfs[yLabel] = pd.DataFrame(metricList)
            dfs[yLabel].columns = [yLabel, 'index']

    return dfs


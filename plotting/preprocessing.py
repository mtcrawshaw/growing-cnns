import os
import json

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms

#DATA PREPROCESSING

def read_log(filename, lengths=None):

    # Read and parse log into data frame
    with open(filename, encoding='utf-8') as resultsFile:
        results = json.load(resultsFile)

    #===DEBUG===
    print(results)
    #===DEBUG=== 

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
            #===DEBUG===
            print("Enters performanceMetrics loop.")
            #===DEBUG=== 
            
            metricList = []
            yLabel = '%s_%s' % (split, metric)
            for i in range(numIterations):
                row = []
                row.append(results[splitKey][i][metric])
                row.append(i * printFrequency)

                metricList.append(list(row))

            if lengths is None:
                #===DEBUG===
                print("lengths is None.")
                #===DEBUG===     
                continue

            # Extend the dataframe length so that all dataframes which will
            # be plotted together have the same length
            lastValue = results[splitKey][numIterations - 1][metric]
            i = numIterations - 1
            while len(metricList) < lengths[split]:
                i += 1
                row = []
                row.append(lastValue)
                row.append(i * printFrequency)
                metricList.append(list(row))

            #===DEBUG===
            print("yLabel", yLabel)
            #===DEBUG=== 
            
            dfs[yLabel] = pd.DataFrame(metricList)
            dfs[yLabel].columns = [yLabel, 'index']

    return dfs

def getLogLengths(experimentNames):
    lengths = {'train': 0, 'validate': 0}

    projectRoot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for experimentName in experimentNames:
        logFilename = os.path.join(projectRoot, 'experiments',
                experimentName, '%s.log' % experimentName)

        with open(logFilename, 'r') as f:
            results = json.load(f)

        for split in lengths.keys():
            splitKey = '%sResults' % split
            lengths[split] = max(lengths[split], len(results[splitKey]))

    return lengths

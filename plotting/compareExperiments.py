import os
import math
import argparse

# To handle running this script as main, or just import this script
try:
    from .plotSettings import *
    from .preprocessing import read_log
    from .graph import graph
except:
    from plotSettings import *
    from preprocessing import read_log
    from graph import graph

def main(**args):
 
    # Read logs
    projectRoot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dfs = {}
    for i, experimentName in enumerate(args['experimentNames']):
        logFilename = os.path.join(projectRoot, 'experiments',
                experimentName, '%s.log' % experimentName)
        currentDfs = read_log(logFilename)

        # Rename columns in dfs to avoid overlap
        for metric in currentDfs.keys():
            columns = {}
            for column in currentDfs[metric].columns:
                if column == 'index':
                    continue

                columns[column] = "%d_%s" % (i, column)

            currentDfs[metric] = currentDfs[metric].rename(index=str,
                    columns=columns)

        # Build up list of dfs across experiments
        for metric, df in currentDfs.items():
            if metric in dfs.keys():
                dfs[metric] = dfs[metric].merge(
                    df.set_index('index'),
                    left_on='index',
                    right_on='index'
                )
            else:
                dfs[metric] = df

    # Create plot
    comparisonDir = os.path.join(projectRoot, 'plotting', 'comparisonPlots')
    if not os.path.isdir(comparisonDir):
        os.makedirs(comparisonDir)

    comparisonName = 'comparison'
    for experimentName in args['experimentNames']:
        comparisonName += '_%s' % experimentName

    plotFile = os.path.join(comparisonDir, '%s.svg' % comparisonName)
    graph(dfs, comparisonName, plotFile)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Growing CNNs with PyTorch. \
            Plot the results of two different experiments on the same plot.')
    parser.add_argument('experimentNames', type=str, help='Names of \
            experiments whose results to plot.')
    args = parser.parse_args()
    args = vars(args)

    args['experimentNames'] = args['experimentNames'].split(',')
    main(**args)


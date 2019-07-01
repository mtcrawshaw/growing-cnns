import os
import math
import argparse

# To handle running this script as main, or just import this script
try:
    from .plotSettings import *
    from .preprocessing import *
    from .graph import graph
except:
    from plotSettings import *
    from preprocessing import *
    from graph import graph

def main(**kwargs):
 
    # Read log
    projectRoot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    logFilename = os.path.join(projectRoot, 'experiments',
            kwargs['experimentName'], '%s.log' % kwargs['experimentName'])
    lengths = getLogLengths([logFilename])
    dfs = read_log(logFilename, lengths)

    #===DEBUG===
    print(dfs)
    # exit()
    #===DEBUG===

    # Create plot
    plotFile = os.path.join(projectRoot, 'experiments', kwargs['experimentName'],
            '%s.svg' % kwargs['experimentName'])
    graph(dfs, kwargs['experimentName'], plotFile)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Growing CNNs with PyTorch. \
            Plot the result of an experiment.')
    parser.add_argument('experimentName', type=str, help='Name of experiment \
            whose results to plot.')
    args = parser.parse_args()

    args = vars(args)
    main(**args)


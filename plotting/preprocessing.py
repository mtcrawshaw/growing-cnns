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

    numIterations = min(len(results['trainResults']),
            len(results['validateResults']))

    # This is kind of hacky, but it's the quickest way to access
    # the print frequency of training without passing it here
    printFrequency = results['trainResults'][1]['iteration']

    metricList = []
    for i in range(numIterations):
        row = []

        iteration = i * printFrequency
        row.append(iteration)
        for split in splits:
            splitKey = '%sResults' % split

            for metric in performanceMetrics:
                row.append(results[splitKey][i][metric])

        metricList.append(list(row))

    metricValues = pd.DataFrame(metricList)
    yLabels = []
    for split in splits:
        for metric in performanceMetrics:
            yLabels.append('%s_%s' % (split, metric))
    metricValues.columns = ['index'] + yLabels

    # Create data frames for each subplot
    dfs = []
    for metric in yLabels:
        dfs.append(metricValues[[metric, 'index']])

    return dfs, yLabels

def create_subplot(**kwargs):

    ax = kwargs['ax'] 
    xaxis = kwargs['xaxis'] 
    yaxis = kwargs['yaxis'] 
    df = kwargs['df'] 
    ylabel = kwargs['ylabel'] 
    color_index = kwargs['color_index'] 
    num_colors = kwargs['num_colors']
    xlabel = kwargs['xlabel']
    y_axis_label_size = kwargs['y_axis_label_size']
    x_axis_label_size = kwargs['x_axis_label_size']
    legend_size = kwargs['legend_size'] 
    tick_label_size = kwargs['tick_label_size']
    axis_font = kwargs['axis_font']
    legend_font = kwargs['legend_font']
    text_opacity = kwargs['text_opacity']
    xaxis_opacity = kwargs['xaxis_opacity']
    
    #===PLOT===
    graph = df.plot(x=xaxis, 
                    y=yaxis,
                    ax=ax, 
                    use_index=True)
                    #legend=True)
    # john
    plt.legend(loc='best')

    # distinct line colors/styles for many lines
    #LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    cm = plt.get_cmap('magma') #'gist_rainbow'
    
    plt.gca().get_lines()[0].set_color(cm(color_index//NUM_STYLES*float(NUM_STYLES)/num_colors))
    plt.gca().get_lines()[0].set_linewidth(3.0)
    color_index += 1

    # add axis labels
    plt.xlabel(xlabel, 
               fontproperties=axis_font, 
               fontsize = y_axis_label_size, 
               alpha=text_opacity)
    plt.ylabel(ylabel, 
               fontproperties=axis_font, 
               fontsize = y_axis_label_size, 
               alpha=text_opacity)

    # change font of legend
    L = graph.legend(prop={'size': legend_size})
    plt.setp(L.texts, fontproperties=legend_font, alpha=text_opacity)

    # set size of tick labels
    graph.tick_params(axis = 'both', 
                      which = 'major', 
                      labelsize = tick_label_size)

    # set fontname for tick labels
    for tick in graph.get_xticklabels():
        tick.set_fontname("DecimaMonoPro")
    for tick in graph.get_yticklabels():
        tick.set_fontname("DecimaMonoPro")

    # set color for tick labels
    [t.set_color('#303030') for t in ax.xaxis.get_ticklabels()]
    [t.set_color('#303030') for t in ax.yaxis.get_ticklabels()]

    # create bolded x-axis
    graph.axhline(y = 0, # 0
                  color = 'black', 
                  linewidth = 1.3, 
                  alpha = xaxis_opacity)

    # Set color of subplots. 
    ax.set_facecolor('#F0F0F0')
     
    return graph, color_index

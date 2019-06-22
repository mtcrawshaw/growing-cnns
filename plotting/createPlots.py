import io
import os
import sys
import csv
import math
import json
import pprint
import argparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms

import pandas as pd
import numpy as np

#DATA PREPROCESSING

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def create_subplot(ax, xaxis, yaxis, df, ylabel, column_total, color_index, NUM_COLORS):

    #===PLOT===
    graph = df.plot(x=xaxis,
                    y=yaxis,
                    ax=ax,
                    use_index=True)
                    #legend=True)
    # john
    plt.legend(loc='best')

    # hacks
    # ax.set_ylim(top=0.93, bottom=0.4)
    # ax.set_xlim(left=-0.1, right=10.5)

    MARKERS=['.',',','o','v','s','p','P','H','+','x','X','D','d','|','_','<','>','^','8','*','h','1','2','3','4']



    # distinct line colors/styles for many lines
    #LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)

    use_markers = False
    if use_markers:
        NUM_MARKERS = len(MARKERS)
        assert len(MARKERS) >= column_count
    cm = plt.get_cmap('magma') #'gist_rainbow'

    j = 0
    while color_index < column_total:
        plt.gca().get_lines()[j].set_color(cm(color_index//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))

        if use_markers:
            plt.gca().get_lines()[j].set_marker(MARKERS[j])
            # plt.gca().get_lines()[j].set_linestyle(LINE_STYLES[i%NUM_STYLES])
            plt.gca().get_lines()[j].set_markersize(7.0)

        plt.gca().get_lines()[j].set_linewidth(3.0)
        color_index += 1
        j += 1

    # add axis labels
    plt.xlabel(xlabel,
               fontproperties=prop3,
               fontsize = y_axis_label_size,
               alpha=text_opacity)
    plt.ylabel(ylabel,
               fontproperties=prop3,
               fontsize = y_axis_label_size,
               alpha=text_opacity)

    # change font of legend
    L = graph.legend(prop={'size': legend_size})
    plt.setp(L.texts, fontproperties=prop4, alpha=text_opacity)

    # set size of tick labels
    graph.tick_params(axis = 'both',
                      which = 'major',
                      labelsize = tick_label_size)

    # set fontname for tick labels
    for tick in graph.get_xticklabels():
        tick.set_fontname("DecimaMonoPro")
    for tick in graph.get_yticklabels():
        tick.set_fontname("DecimaMonoPro")

    # set column_labels
    # graph.set_xticklabels(row_labels)
    # plt.xticks(np.arange(0, num_xtick_labels, 1.0),rotation=60)

    # set color for tick labels
    [t.set_color('#303030') for t in ax.xaxis.get_ticklabels()]
    [t.set_color('#303030') for t in ax.yaxis.get_ticklabels()]

    # create bolded x-axis
    graph.axhline(y = 0, # 0
                  color = 'black',
                  linewidth = 1.3,
                  alpha = xaxis_opacity)

    return graph, color_index


def main(args):

    # Get log
    projectRoot = os.path.abspath(os.path.join('..', 
        os.path.dirname(__file__)))
    experimentDir = os.path.join(projectRoot, 'experiments',
            args.experimentName)
    logFilename = os.path.join(experimentDir, '%s.log' % args.experimentName)
    settingsFilename = os.path.join(experimentDir, '%s_settings.json' %
            args.experimentName)

    # Read in log
    rows = []
    with open(logFilename, encoding='utf-8') as json_file:
        json_data = json.load(json_file)

        train_list = []
        train_keys = []
        trainlog_list = json_data['trainResults']
        eval_list = []
        eval_keys = []
        evallog_list = json_data['validateResults']

        # Construct train array.
        for i, stepdict in enumerate(trainlog_list):
            row = []
            for key, val in stepdict.items():
                if i == 0:
                    train_keys.append(key)
                if isint(val):
                    row.append(int(val))
                elif isfloat(val):
                    row.append(float(val))
                else:
                    row.append(val)
            train_list.append(row)

        print(train_keys)

        # Construct eval array.
        for i, stepdict in enumerate(evallog_list):
            row = []
            for key, val in stepdict.items():
                if i == 0:
                    eval_keys.append(key)
                if isint(val):
                    row.append(int(val))
                elif isfloat(val):
                    row.append(float(val))
                else:
                    row.append(val)

            eval_list.append(row)

        train = pd.DataFrame(train_list)
        evalu = pd.DataFrame(eval_list)
        train.columns = train_keys
        evalu.columns = eval_keys
        train['index'] = train.index
        evalu['index'] = evalu.index
        train['index'] = train['index'].apply(lambda x: x*10)
        evalu['index'] = evalu['index'].apply(lambda x: x*10)

    dfs = []

    print(train_keys)

    # Handle column labels.
    ylabels = ['epoch', 'iteration', 'time', 'loss', 'top1/top5', 'growthStep']
    column_counts = [1,1,1,1,2,1]

    ylabels = []
    for i,count in enumerate(column_counts):
        if count == 1:
            ylabels.append(train_keys[i])
        else:
            y_label = ""
            for j in range(count):
                print(ylabel)
                print(ylabel + train_keys[i + j])
                y_label = ylabel + train_keys[i + j] + "/"
            ylabels.append(ylabel[:-2])
    print(ylabels)

    loss = train[['loss', 'index']]
    time = train[['time', 'index']]
    tops = train[['top1', 'top5', 'index']]
    growthStep = train[['growthStep', 'index']]
    iteration = train[['iteration', 'index']]
    epoch = train[['epoch', 'index']]
    dfs.append(epoch)
    dfs.append(iteration)
    dfs.append(time)
    dfs.append(loss)
    dfs.append(tops)
    dfs.append(growthStep)


    # PLOTTING

    #================================================
    # PARAMETERS
    #================================================

    # size of ENTIRE PLOT
    plot_height = 20 # 7.25
    plot_width = 20
    num_empty_ticks = 0
    num_xtick_labels = len(row_labels) + num_empty_ticks
    #assert num_xtick_labels > len(row_labels)

    # dataframe
    df = scores

    # x-axis
    xaxis = 'index'

    # y-axis
    yaxis = None

    # text
    title_text =  filename
    subtitle_text = "train"
    xlabel = ""
    ylabel = "Loss"
    banner_text = "©craw"

    # edges of plot in figure (padding)
    top = 0.90
    bottom = 0.1 #0.18 -- old
    left = 0.08 # 0.1 -- old
    right = 0.96

    # change title_pad to adjust xpos of title in pixels
    # + is left, - is right
    title_pad_x = 0

    # Title sizes
    title_pos_y = 0.95
    subtitle_pos_y = 0.92
    title_fontsize = 50
    subtitle_fontsize = 30

    # opacity
    text_opacity = 0.75
    xaxis_opacity = 0.7

    # sizing
    tick_label_size = 14
    legend_size = 14
    y_axis_label_size = 14
    x_axis_label_size = 24
    banner_text_size = 14

    # import font
    prop = fm.FontProperties(fname='DecimaMonoPro.ttf')
    prop2 = fm.FontProperties(fname='apercu_medium_pro.otf')
    prop3 = fm.FontProperties(fname='Apercu.ttf')
    prop4 = fm.FontProperties(fname='Apercu.ttf', size=legend_size)

    #ticks_font = matplotlib.font_manager.FontProperties(family='DecimaMonoPro', style='normal', size=12, weight='normal', stretch='normal')

    #================================================
    # END OF PARAMETERS
    #================================================




    # =========================================================

    # figure initialization
    fig, axlist = plt.subplots(figsize=(plot_width, plot_height),nrows=len(dfs))
    color_index = 0
    column_total = 0
    NUM_COLORS = sum(column_counts)

    for i, df in enumerate(dfs):
        ax = axlist[i]
        plt.sca(ax)
        style.use('fivethirtyeight')
        column_total += column_counts[i]
        graph, color_index = create_subplot(ax, xaxis, yaxis, df, ylabels[i], column_total, color_index, NUM_COLORS)



    xlabel = "Iterations"
    # add axis labels
    plt.xlabel(xlabel,
               fontproperties=prop3,
               fontsize = 24,
               alpha=text_opacity)

    # =========================================================

    # transforms the x axis to figure fractions, and leaves y axis in pixels
    xfig_trans = transforms.blended_transform_factory(fig.transFigure, transforms.IdentityTransform())
    yfig_trans = transforms.blended_transform_factory(transforms.IdentityTransform(), fig.transFigure)

    # banner positioning
    banner_y = math.ceil(banner_text_size * 0.6)

    # banner text
    banner = plt.annotate(banner_text,
             xy=(0.02, banner_y*0.8),
             xycoords=xfig_trans,
             fontsize = banner_text_size,
             color = '#FFFFFF',
             fontname='DecimaMonoPro')

    # banner background height parameters
    pad = 2 # points
    bb = ax.get_window_extent()
    print("bbHeight", bb.height)
    h = bb.height/fig.dpi
    h = h * len(column_counts)
    height = ((banner.get_size()+2*pad)/72.)/h
    # height = 0.01
    print("height", height)

    # banner background
    rect = plt.Rectangle((0,0),
                         width=1,
                         height=height,
                         transform=fig.transFigure,
                         zorder=3,
                         fill=True,
                         facecolor="grey",
                         clip_on=False)
    ax.add_patch(rect)

    #transform coordinate of left
    display_left_tuple = xfig_trans.transform((left,0))
    display_left = display_left_tuple[0]

    # shift title
    title_shift_x = math.floor(tick_label_size * 2.6)
    title_shift_x += title_pad_x

    # title
    graph.text(x = display_left - title_shift_x, y = title_pos_y,
               transform = yfig_trans,
               s = title_text,
               fontproperties = prop2,
               weight = 'bold',
               fontsize = title_fontsize,
               alpha = text_opacity)

    # subtitle, +1 accounts for font size difference in title and subtitle
    graph.text(x = display_left - title_shift + 1, y = subtitle_pos_y,
               transform = yfig_trans,
               s = subtitle_text,
               fontproperties=prop3,
               fontsize = subtitle_fontsize,
               alpha = text_opacity)


    # adjust position of subplot in figure
    plt.subplots_adjust(top=top)
    plt.subplots_adjust(bottom=bottom)
    plt.subplots_adjust(left=left)
    plt.subplots_adjust(right=right)

    # save to .svg
    plt.savefig(filename_no_extension + ".svg", dpi=300)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create plots from experiment \
            logs')
    parser.add_argument('experimentName', help='Name of experiment. Should \
            match the name of the experiment directory in experiments/')
    args = parser.parse_args()

    main(args)

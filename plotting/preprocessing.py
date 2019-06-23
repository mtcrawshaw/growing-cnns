import json

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms

#DATA PREPROCESSING

def read_log(filename, phase):

    phases = ['train', 'validate', 'test']
    assert phase in phases
    
    rows = []
    filename_no_extension = filename.split('.')[0]
    print("Reading from file:", filename)

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

    with open(filename, encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        
        phase_key = phase + "Results"

        row_list = []
        log_list = json_data[phase_key]

        # Construct train array. 
        for i, stepdict in enumerate(log_list):
            row = []
            for key, val in stepdict.items():
                if isint(val):
                    row.append(int(val))
                elif isfloat(val):
                    row.append(float(val))
                else:
                    row.append(val)
            row_list.append(row) 

        log_df = pd.DataFrame(log_list)
        keys = list(log_df.columns)
        log_df['index'] = log_df.index
        log_df['index'] = log_df['index'].apply(lambda x: x*10)
       
    dfs = []

    # This column_counts generation process assumes 
    # `top5` always follows `top1` in keys. 
    column_counts = []
    i = 0
    for i in range(len(keys)):
        key = keys[i]
        if key == 'top1':
            column_counts.append(2)
        elif key != 'top5':
            column_counts.append(1)
        i += 1

    """
    # Handle distribution of columns in subplots.
    if phase == 'train': 
        column_counts = [1,1,1,1,2,1]
        try: 
            assert sum(column_counts) == len(keys)
        except AssertionError:
            column_counts = [1,1,1,1,2]
            assert sum(column_counts) == len(keys) 
    else:
        column_counts = [1,1,2,1]
    try: 
        assert sum(column_counts) == len(keys)
    except AssertionError:
        column_counts = [1,1,2]
        assert sum(column_counts) == len(keys) 
    """

    # Iterate over column split, and create a seperate DataFrame for 
    # each subplot. Add the subplot names to `ylabels`. 
    ylabel = ""
    ylabels = []
    for i,count in enumerate(column_counts):
        key_list = ['index']
        if count == 1:
            ylabels.append(keys[i])
            key_list.append(keys[i])
            dfs.append(log_df[key_list])
        else:
            words = []
            for j in range(count):
                words.append(keys[i + j])
                words.append("/")
                key_list.append(keys[i + j])
            ylabels.append("".join(words[:-1]))
            dfs.append(log_df[key_list])
    print("Generating", len(dfs), "subplots.")
    
    return dfs, ylabels, column_counts

def create_subplot(**kwargs):

    ax = kwargs['ax'] 
    xaxis = kwargs['xaxis'] 
    yaxis = kwargs['yaxis'] 
    df = kwargs['df'] 
    ylabel = kwargs['ylabel'] 
    column_total = kwargs['column_total'] 
    color_index = kwargs['color_index'] 
    NUM_COLORS = kwargs['NUM_COLORS']
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

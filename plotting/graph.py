import math
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms
import pandas as pd

# To handle running this script as main, or just import this script
try:
    from .plotSettings import *
except:
    from plotSettings import *

def graph(dfs, title, filename):
    
    # figure initialization
    fig, axlist = plt.subplots(figsize=(plot_width, plot_height),nrows=len(dfs))
    color_index = 0
    num_colors = sum([len(df.columns) - 1 for df in dfs.values()])
    
    for i, (metric, df) in enumerate(dfs.items()):
        ax = axlist[i]
        plt.sca(ax)
        style.use('fivethirtyeight')
        graph, color_index = create_subplot(
                ax=ax, 
                xaxis=xaxis, 
                yaxis=yaxis, 
                df=df, 
                ylabel=metric, 
                color_index=color_index, 
                num_colors=num_colors,
                xlabel=xlabel,
                y_axis_label_size=y_axis_label_size,
                x_axis_label_size=x_axis_label_size,
                legend_size=legend_size, 
                tick_label_size=tick_label_size,
                axis_font=prop,
                legend_font=prop,
                text_opacity=text_opacity,
                xaxis_opacity=xaxis_opacity,
        )

    newXLabel = "Iterations"
    # add axis labels  
    plt.xlabel(newXLabel, 
               fontproperties=prop, 
               fontsize = 24, 
               alpha=text_opacity)

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
    h = bb.height/fig.dpi
    h = h * len(dfs)
    height = ((banner.get_size()+2*pad)/72.)/h
    # height = 0.01

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
               s = title,
               fontproperties = prop,
               weight = 'bold', 
               fontsize = title_fontsize,
               alpha = text_opacity)

    # subtitle, +1 accounts for font size difference in title and subtitle
    graph.text(x = display_left - title_shift_x + 1, y = subtitle_pos_y, 
               transform = yfig_trans,
               s = '',
               fontproperties=prop,
               fontsize = subtitle_fontsize, 
               alpha = text_opacity)


    # adjust position of subplot in figure
    plt.subplots_adjust(top=top)
    plt.subplots_adjust(bottom=bottom)
    plt.subplots_adjust(left=left)
    plt.subplots_adjust(right=right)

    plt.savefig(filename, dpi=300)


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
    
    cm = plt.get_cmap('prism')
    
    for j in range(len(df.columns) - 1):
        plt.gca().get_lines()[j].set_color(cm(color_index/num_colors))
        plt.gca().get_lines()[j].set_linewidth(3.0)
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

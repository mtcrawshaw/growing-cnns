import matplotlib
matplotlib.use('agg')
import matplotlib.font_manager as fm

# Size of ENTIRE PLOT. 
plot_height = 20 # 7.25
plot_width = 20
num_empty_ticks = 0

# x-axis.
xaxis = 'index'

# y-axis.
yaxis = None

# Text. 
xlabel = ""
banner_text = "©craw"

# Set edges of plot in figure (padding). 
top = 0.90
bottom = 0.1 #0.18 -- old
left = 0.08 # 0.1 -- old
right = 0.96

# Title sizes. 
title_pad_x = 0     # + is left, - is right
title_pos_y = 0.95
subtitle_pos_y = 0.92
title_fontsize = 50
subtitle_fontsize = 30

# Opacity.
text_opacity = 0.75
xaxis_opacity = 0.7

# Sizing.
tick_label_size = 14
legend_size = 14
y_axis_label_size = 14
x_axis_label_size = 24
banner_text_size = 14

# Import font. 
prop = fm.FontProperties(fname='DecimaMonoPro.ttf')

#ticks_font = matplotlib.font_manager.FontProperties(family='DecimaMonoPro', style='normal', size=12, weight='normal', stretch='normal')


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


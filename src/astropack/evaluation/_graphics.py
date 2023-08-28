import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

from math import log10, floor

label_fontdict = {'family': 'sans-serif',
                  'weight': 'normal',
                  'size': 16}

legend_label_fontdict = {'family': 'sans-serif',
                         'weight': 'normal',
                         'size': 14}

legend_title_fontdict = {'family': 'sans-serif',
                         'weight': 'bold',
                         'size': 18}

def plot_test_graphs(predictions, true_values, bins, cmap_name, param_string, param_unit, n_ticks, legend):

    if not bins:
        bins = [min(true_values) - 1, max(true_values) + 1]

    df = pd.merge(left = true_values, left_index = True,
                  right = predictions, right_index = True)

    df.columns = ['TRUE_VALUE', 'PREDICTION']
    df['ERROR'] = df['PREDICTION'] - df['TRUE_VALUE']

    fig, ax = plt.subplots(nrows = 1, ncols = 2,
                           width_ratios = [0.6, 0.4],
                           figsize = (15, 6))

    colors = mpl.colormaps[cmap_name](np.linspace(0.5, 1.00, len(bins) - 1))
    
    bins_intervals = []
    for bin_index in range(0, len(bins) - 1):
        bin_min = bins[bin_index]
        bin_max = bins[bin_index + 1]
        bins_intervals.append(f'[{bin_min} {param_unit}, {bin_max} {param_unit}]')

        df_bin = df[(df['PREDICTION'] >= bin_min) & (df['PREDICTION'] < bin_max)].copy()

        sns.scatterplot(data = df_bin, x = 'PREDICTION', y = 'TRUE_VALUE',
                        ax = ax[0], s = 9, color = colors[bin_index],
                        linewidth = 0, zorder = 2)
        sns.kdeplot(data = df_bin, x = 'ERROR', ax = ax[1],
                    color = colors[bin_index])
        
    f = lambda m,c: ax[0].plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(len(bins_intervals))]

    min_lim = round_to_1(bins[0] - (bins[-1] - bins[0]) * 0.05)
    max_lim = round_to_1(bins[-1] + (bins[-1] - bins[0]) * 0.05)

    ax[0].plot([min_lim, max_lim],
               [min_lim, max_lim],
               ls = '--',
               lw = 1.5,
               color = 'k',
               zorder = 3)

    ax[0] = beautify_graph(ax = ax[0],
                           x_limits = [min_lim, max_lim],
                           y_limits = [min_lim, max_lim],
                           n_ticks = n_ticks,
                           x_label = f'Predicted {param_string}',
                           y_label = f'True {param_string}')


    min_lim = round_to_1(-(df['ERROR'].abs().median() * 20))
    max_lim = round_to_1((df['ERROR'].abs().median() * 20))

    ax[1] = beautify_graph(ax = ax[1],
                           x_limits = [min_lim, max_lim],
                           y_limits = None,
                           n_ticks = n_ticks,
                           x_label = f'Error',
                           y_label = f'Density')

    ax[1].tick_params(axis = 'y', labelsize = 0)
    
    if legend:
        leg = fig.legend(handles,
                        bins_intervals,
                        title = f'{param_string}',
                        title_fontproperties = legend_title_fontdict,
                        ncols = len(bins_intervals),
                        loc = 'upper center',
                        bbox_to_anchor = (0.5, -0.075),
                        framealpha = 1,
                        prop = legend_label_fontdict,
                        markerscale = 3,
                        borderpad = 1)
        
        leg._legend_box.sep = 20
    
    return fig


def beautify_graph(ax, x_limits, y_limits, n_ticks, x_label, y_label):
    if x_limits:
        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_xticks(np.linspace(x_limits[0],
                                  x_limits[1],
                                  n_ticks))
        
    if y_limits:
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_yticks(np.linspace(y_limits[0],
                                  y_limits[1],
                                  n_ticks))

    ax.tick_params(labelsize = 14)

    ax.set_xlabel(x_label,
                  fontdict = label_fontdict,
                  labelpad = 15)
    ax.set_ylabel(y_label,
                  fontdict = label_fontdict,
                  labelpad = 15)

    ax.grid(zorder = 0)
    
    return ax


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
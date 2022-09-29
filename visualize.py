import pandas as pd
import numpy as np
import os
import glob

import matplotlib
import matplotlib.pyplot as plt

font = 'Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.it'] = font
plt.rcParams['mathtext.bf'] = font
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5

linewidth = 2.5

# import tensorflow as tf
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
print("TensorBoard version: ", tb.__version__)

PINK = (247/255, 112/255, 136/255)
GREEN = (51/255, 176/255, 122/255)
BLUE = (128/255, 150/255, 244/255)
BLUEBLUE = (0, 83/255, 214/255)
YELLOW = (255/255, 161/255, 0/255)
BLACK = (0, 0, 0)

# https://yeun.github.io/open-color/#red
VIOLET9 = (95/255, 61/255, 196/255)
PINK9 = (166/255, 30/255, 77/255)
GRAY9 = (33/255, 37/255, 41/255)

GRAY8 = (52/255, 58/255, 64/255)

GRAY7 = (73/255, 80/255, 87/255)
ORANGE7 = (247/255, 103/255, 7/255)

GRAY6 = (134/255, 142/255, 150/255)

GRAY4 = (206/255, 212/255, 218/255)

RED4 = (255/255, 135/255, 135/255)
PINK4 = (247/255, 131/255, 172/255)
GRAPE4 = (218/255, 119/255, 242/255)
VIOLET4 = (151/255, 117/255, 250/255)
INDIGO4 = (116/255, 143/255, 252/255)
BLUE4 = (77/255, 171/255, 247/255)
CYAN4 = (59/255, 201/255, 219/255)
TEAL4 = (56/255, 217/255, 169/255)
GREAN4 = (105/255, 219/255, 124/255)
LIME4 = (169/255, 227/255, 75/255)
YELLOW4 = (255/255, 212/255, 59/255)
ORANGE4 = (255/255, 169/255, 77/255)

# COLOR_LIST = [GRAY7, GRAPE4, VIOLET4, BLUE4, TEAL4, LIME4, YELLOW4, ORANGE4, RED4]
COLOR_LIST = [RED4, ORANGE4, YELLOW4, LIME4, TEAL4, INDIGO4, VIOLET4, GRAPE4, GRAY7, PINK, GREEN, BLUE, YELLOW, BLACK]
# COLOR_LIST = [GRAY7, VIOLET4, RED4, TEAL4, YELLOW4, GRAPE4, LIME4, BLUE4, ORANGE4]


def load_df_from_tb_event(tb_event, col='evaluation/average_returns'):
    ea = event_accumulator.EventAccumulator(tb_event)
    ea.Reload()
    try:
        df = pd.DataFrame(ea.Scalars(col))
    except:
        print(f"tb_event: {tb_event}")
        raise
    return df[['step', 'value']]


def get_data_from_all_seeds(tb_file_list, col='evaluation/avearge_returns', window=1):
    df = None
    for tb_file in tb_file_list:
        if df is None:
            # Dirty and quick fix to incorporate 
            # for csv data from KH (eval every 10000) 
            # and tensorboard log from JS (eval every 40000).
            try:
                df = pd.read_csv(tb_file)
                df = df.rename(columns={'Step': 'step', 'Value': 'value'})
                df = df[['step', 'value']]
                df = df[df.index % window == 0]
            except:
                df = load_df_from_tb_event(tb_file, col=col)
        else:
            try:
                append_df = pd.read_csv(tb_file)
                append_df = append_df.rename(columns={'Step': 'step', 'Value': 'value'})
                append_df = append_df[['step', 'value']]
                df = pd.concat([df, append_df], axis=1)
                df = df[df.index % window == 0]
            except:
                df = pd.concat([df, load_df_from_tb_event(tb_file, col=col)], axis=1)
    return df


def exp_smooth(df, alpha=0.4):
    return df['value'].ewm(alpha=alpha).mean()


def rolling(df, window=4):
    return df['value'].rolling(window, min_periods=1).mean()
    
    
def mean_std(df):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    return df_mean, df_mean - df_std, df_mean + df_std


def process_data(tb_list, col='evaluation/average_returns', verbose=True, window=1):
    df_list = get_data_from_all_seeds(tb_list, col=col, window=window)
    if verbose:
        print(df_list)
    smoothed_mean, smoothed_under_std, smoothed_over_std = mean_std(rolling(df_list, window=window))
    
    x = df_list['step'].iloc[:, 1].to_numpy()
    
    y_mean = smoothed_mean.to_numpy()
    y_under_std = smoothed_under_std.to_numpy()
    y_over_std = smoothed_over_std.to_numpy()
    return x, y_mean, y_under_std, y_over_std


def draw_graph(title='',
               xlim_lower=0,
               xlim_upper=1000000, 
               ylim_upper=100,
               ylim_lower=0,
               fill_density=0.15,
               figsize=(5, 3.5),
               idx=201,
               verbose=False,
               no_legend=False,
               save=True,
               save_path='./graphs/',
               show_title=True,
               show_var=True,
               legend_loc='upper left',
               color_list=COLOR_LIST,
               col='evaluation/average_returns',
               extension='png',
               **kwargs,
              ):
    line_num = 0
    label_list = []

    xticks = np.linspace(xlim_lower, xlim_upper, 5)
    yticks = np.linspace(ylim_lower, ylim_upper, 5)

    for key, value in kwargs.items():
        if 'label' in key:
            label_list.append(value)
    
    fill_density = fill_density
    _, ax = plt.subplots(1, 1, figsize=figsize, dpi=500)
    
    for key, value in kwargs.items():
        if 'tb_list' in key:
            xx, yy_mean, yy_under_std, yy_over_std = process_data(value, col=col, verbose=verbose)
            ax.plot(xx[:idx], yy_mean[:idx], color=color_list[line_num], label=label_list[line_num], linewidth=linewidth * 1.25)
            if show_var:
                ax.fill_between(xx[:idx], yy_under_std[:idx], yy_over_std[:idx], facecolor=(*color_list[line_num], fill_density), edgecolor=(0, 0, 0, 0))
            print(f"{label_list[line_num]}: {yy_mean[-1]:.4f} ± {yy_mean[-1] - yy_under_std[-1]:.4f}")
            line_num += 1
    
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Average Return', fontsize=14)
    if show_title:
        ax.set_title(title, fontsize=16)

    ax.grid(alpha=1.0, linestyle=':', linewidth=0.25)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_yticks(yticks)
    
    ax.set_xticks(xticks)
    ax.set_xticks([100000, 300000, 500000, 700000, 900000], minor=True)

    def set_xtick(x, p):
        return '{}$\\times 10^5$'.format(int(x / 100000))
    # NOTE: use xtick with 10^4 or xlabel with 10^4
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(set_xtick)
    )
    ax.xaxis.major.formatter._useMathText = True

    ax.set_xlim(xlim_lower, xlim_upper)
    ax.set_ylim(ylim_lower, ylim_upper)

    if not no_legend:
        leg = ax.legend(fancybox=False, fontsize=8, edgecolor='black', borderaxespad=0.1, handlelength=1.5, loc=legend_loc)
        leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    
    if save:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + '/' + title + f".{extension}")
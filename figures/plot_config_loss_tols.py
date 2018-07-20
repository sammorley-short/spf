# Import Python packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set matplotlib font and padding defaults
font = {'family': 'serif',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['axes.titlepad'] = 20
matplotlib.rcParams['axes.labelpad'] = 10


def plot_n_config_loss_tol(in_files, styles, out_file, norm=False):
    """
        Plots GPF and SPF teleportation channels' confguration loss tolerances.
        Line colors, styles and labels should be provided in a styles dict.
        If norm, the total number of confirgurations is normalised to one
        Adds title and outputs PNG to out_file
    """
    width = 0.666
    # Initialise figure
    plt.figure()
    fig, ax = plt.subplots()
    # Plots input datasets
    for i, (in_file, style) in enumerate(zip(in_files, styles)):
        # Reads in CSV data into pandas dataframe object
        df = pd.read_csv(in_file, index_col=[0])
        # Creates x- and y-value arrays and their associated errors
        x_vals = np.array(df.index.values)
        y_vals = np.array(df.config_tol) if norm else np.array(df.tol_configs)
        if i == 0:
            # Defines y_max and x axis limits and plots max configs
            y_max = y_vals * 0 + 1 if norm else np.array(df.tot_configs)
            ax.bar(x_vals, y_max, width, alpha=1,
                   color='lightgray')
            x_min = min(x_vals) - 1 + width/2
            x_max = max(x_vals) + 1 - width/2
        # Plots SPF and GPF bars
        ax.bar(x_vals, y_vals, width,
               alpha=1, color=style['color'], label=style['label'])
    # Sets plot limits and labels for abs. or norm. versions
    if norm:
        ax.set_ylim(0, 1)
        ylabel = 'Loss-tolerant $n_l$-qubit configurations (norm.)'
        loc = 'upper right'
        y_minor_ticks = np.linspace(0, 1, 11)
        ax.set_yticks(y_minor_ticks, minor=True)
    else:
        ax.set_yscale('log')
        ylabel = 'Loss-tolerant $n_l$-qubit configurations (abs.)'
        loc = 'upper left'
    # Formats axes, adds labels and title
    ax.set_xlim(x_min, x_max)
    plt.xlabel('Lost qubits, $n_l$', size=24)
    plt.ylabel(ylabel, size=24)
    # Creates gridlines
    x_minor_ticks = np.linspace(0, max(x_vals), 1)
    ax.set_xticks(x_minor_ticks, minor=True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.grid(True, ls='--', lw=1, c='k', alpha=0.2, which='major', axis='y')
    ax.grid(True, ls=':', lw=1, c='k', alpha=0.1, which='minor', axis='y')
    # Outputs to file
    fig.set_size_inches(10, 7)
    plt.legend(loc=loc, fontsize=24, framealpha=1)
    plt.savefig(out_file, bbox_inches='tight')


if __name__ == '__main__':
    # =================== PLOTS CONFIGURATION LOSS TOLERANCE ====================
    if not os.path.exists('pngs'):
        os.makedirs('pngs')
    styles = [{'color': 'deepskyblue', 'label': 'SPF'},
              {'color': 'salmon', 'label': 'GPF'}]
    channels = ['W4xL4_SquareLatticeChannel',
                'W4xL4_HexagonalLatticeChannel',
                'W4xL4_TriangularLatticeChannel',
                'B2xD3_TreeToTreeChannel']
    for prefix in channels:
        in_files = ['data/' + prefix + '_MW5_SPF_CONFIG_TOL.csv',
                    'data/' + prefix + '_GPF_CONFIG_TOL.csv']
        out_file = 'pngs/' + prefix + '_MW5_CONFIG_TOL_RATE.png'
        plot_n_config_loss_tol(in_files, styles, out_file, norm=True)
        out_file = 'pngs/' + prefix + '_MW5_CONFIG_TOL_ABS.png'
        plot_n_config_loss_tol(in_files, styles, out_file, norm=False)

    # Plots just SPF for crazy graph
    prefix = 'W4xL4_CrazyGraphChannel'
    in_files = ['data/' + prefix + '_MW5_SPF_CONFIG_TOL.csv']
    styles = [{'color': 'deepskyblue', 'label': 'SPF'}]
    out_file = 'pngs/' + prefix + '_MW5_CONFIG_TOL_RATE.png'
    plot_n_config_loss_tol(in_files, styles, out_file, norm=True)
    out_file = 'pngs/' + prefix + '_MW5_CONFIG_TOL_ABS.png'
    plot_n_config_loss_tol(in_files, styles, out_file, norm=False)

# Import Python packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from textwrap import wrap

# Set matplotlib font and padding defaults
font = {'family': 'serif',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['axes.titlepad'] = 20
matplotlib.rcParams['axes.labelpad'] = 10


def plot_uh_mnt_strategy_comparison(in_files, styles, title, out_file):
    """
        Plots unheralded SPF teleportation rates for a channel for diferrent
        measurement strategies.
        Line colors, styles and labels should be provided in a styles dict.
        Adds title and outputs PNG to out_file
    """
    # Initialise figure
    plt.figure()
    fig, ax = plt.subplots()
    # Plots input datasets
    for in_file, style in zip(in_files, styles):
        # Reads in CSV data into pandas dataframe object
        df = pd.read_csv(in_file, index_col=[0])
        # Creates x- and y-value arrays and their associated errors
        x_vals = np.array(df.index.values)
        y_vals = np.array(df.prob_tel)
        y_errs = np.array(df.prob_tel_std)
        # Plots SPF and GPF lines with errors and adds difference fill
        ax.plot(x_vals, y_vals, color=style['color'], linewidth=1,
                label=style['label'], linestyle=style['line'])
        ax.fill_between(x_vals, y_vals - 1 * y_errs, y_vals + 1 * y_errs,
                        facecolor=style['color'], alpha=0.2, linewidth=0.0)
    # Adds measurement strategy labels
    lgd_title = "\n".join(wrap('Strategy', 12))
    legend = plt.legend(loc='upper right', framealpha=1, title=lgd_title,
                        prop={'weight': 'bold'})
    legend.get_title().set_fontsize('24')  # legend 'Title' fontsize
    plt.setp(legend.get_texts(), fontsize=24)
    # Formats axes and adds labels
    plt.xlabel('Loss probability (per-qubit), $p_l$', size=24)
    plt.ylabel('Teleportation rate, $T$', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    plt.title(title, fontsize=24)
    # Creates gridlines
    minor_ticks = np.linspace(0, 1, 41)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, ls='--', lw=1, c='k', alpha=0.2, which='major')
    ax.grid(True, ls=':', lw=1, c='k', alpha=0.1, which='minor')
    # Outputs to file
    fig.set_size_inches(9, 6)
    plt.savefig(out_file, bbox_inches='tight')


if __name__ == '__main__':
    # ================== PLOTS MNT STRATEGY COMPARISON ======================
    if not os.path.exists('pngs'):
        os.makedirs('pngs')
    in_files = \
        ['data/W4xL4_TriangularLatticeChannel_MW5_100MC_SPF_UH_MT_TEL_RATE.csv',
         'data/W4xL4_TriangularLatticeChannel_MW5_100MC_SPF_UH_MC_TEL_RATE.csv']
    out_file = 'pngs/tri_mnt_strategy_comparison.png'
    styles = [{'color': 'b', 'line': '-', 'label': 'Max-tolerance'},
              {'color': 'r', 'line': '-', 'label': 'Most common'}]
    title = ''

    plot_uh_mnt_strategy_comparison(in_files, styles, title, out_file)

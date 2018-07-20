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


def plot_spf_and_gpf_thresholds(in_files, styles, title, out_file):
    """
        Plots SPF and GPF teleportation lattice channel rates across given
        sizes.
        Line colors, styles and labels should be provided in a styles dict.
        Adds title and outputs PNG to out_file
    """
    # Initialise figure
    plt.figure()
    fig, ax = plt.subplots()
    # Plots input datasets
    for (spf_in_file, gpf_in_file), style in zip(in_files, styles):
        # Reads in CSV data into pandas dataframe object
        spf_df = pd.read_csv(spf_in_file, index_col=[0])
        gpf_df = pd.read_csv(gpf_in_file, index_col=[0])
        # Creates x- and y-value arrays and their associated errors
        spf_x_vals = np.array(spf_df.index.values)
        gpf_x_vals = np.array(gpf_df.index.values)
        spf_y_vals = np.array(spf_df.prob_tel)
        gpf_y_vals = np.array(gpf_df.prob_tel)
        spf_y_errs = np.array(spf_df.prob_tel_std)
        gpf_y_errs = np.array(gpf_df.prob_tel_std)
        # Plots SPF and GPF lines
        ax.plot(spf_x_vals, spf_y_vals, color=style['color'], linewidth=3,
                linestyle='-')
        ax.plot(gpf_x_vals, gpf_y_vals, color=style['color'], linewidth=3,
                linestyle='--')
    # Adds channel and PF line labels
    ch_handles = [Line2D([], [], color=style['color'], linestyle='-',
                         linewidth=2, label=style['label'])
                  for style in styles]
    spf_line = Line2D([], [], color='k', linestyle='-', linewidth=2)
    gpf_line = Line2D([], [], color='k', linestyle='--', linewidth=2)
    pf_handles = [spf_line, gpf_line]
    pf_labels = ['SPF', 'GPF']
    ch_labels = [h.get_label() for h in ch_handles]
    ch_lgd = plt.legend(ch_handles, ch_labels,
                        loc='upper right', fontsize=24, framealpha=1)
    pf_lgd = plt.legend(pf_handles, pf_labels,
                        loc='center right', fontsize=24, framealpha=1)
    ax.add_artist(pf_lgd)
    ax.add_artist(ch_lgd)
    # Formats axes, adds labels and title
    plt.xlabel('Loss probability (per-qubit), $p_l$', size=24)
    plt.ylabel('Teleportation rate, $T$',  size=24)
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
    fig.set_size_inches(8, 7)
    plt.savefig(out_file, bbox_inches='tight')


if __name__ == '__main__':
    # ============== PLOTS CHANNEL THRESHOLDS (SPF and GPF) =================
    if not os.path.exists('pngs'):
        os.makedirs('pngs')
    in_files = \
        [['data/W4xL4_TriangularLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W4xL4_TriangularLatticeChannel_10000MC_GPF_TEL_RATE.csv'],
         ['data/W3xL3_TriangularLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W3xL3_TriangularLatticeChannel_10000MC_GPF_TEL_RATE.csv'],
         ['data/W2xL2_TriangularLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W2xL2_TriangularLatticeChannel_10000MC_GPF_TEL_RATE.csv']]
    styles = [{'color': 'b', 'line': '-', 'label': '$4 \\times 4$'},
              {'color': 'r', 'line': '-', 'label': '$3 \\times 3$'},
              {'color': 'g', 'line': '-', 'label': '$2 \\times 2$'}]
    out_file = 'pngs/tri_SPF_and_GPF_thresholds.png'
    title = 'Triangular Lattice'
    plot_spf_and_gpf_thresholds(in_files, styles, title, out_file)

    in_files = \
        [['data/W4xL4_HexagonalLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W4xL4_HexagonalLatticeChannel_10000MC_GPF_TEL_RATE.csv'],
         ['data/W3xL3_HexagonalLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W3xL3_HexagonalLatticeChannel_10000MC_GPF_TEL_RATE.csv'],
         ['data/W2xL2_HexagonalLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W2xL2_HexagonalLatticeChannel_10000MC_GPF_TEL_RATE.csv']]
    styles = [{'color': 'b', 'line': '-', 'label': '$4 \\times 4$'},
              {'color': 'r', 'line': '-', 'label': '$3 \\times 3$'},
              {'color': 'g', 'line': '-', 'label': '$2 \\times 2$'}]
    out_file = 'pngs/hex_SPF_and_GPF_thresholds.png'
    title = 'Hexagonal Lattice'
    plot_spf_and_gpf_thresholds(in_files, styles, title, out_file)

    in_files = \
        [['data/W4xL4_SquareLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W4xL4_SquareLatticeChannel_10000MC_GPF_TEL_RATE.csv'],
         ['data/W3xL3_SquareLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W3xL3_SquareLatticeChannel_10000MC_GPF_TEL_RATE.csv'],
         ['data/W2xL2_SquareLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
          'data/W2xL2_SquareLatticeChannel_10000MC_GPF_TEL_RATE.csv']]
    styles = [{'color': 'b', 'line': '-', 'label': '$4 \\times 4$'},
              {'color': 'r', 'line': '-', 'label': '$3 \\times 3$'},
              {'color': 'g', 'line': '-', 'label': '$2 \\times 2$'}]
    out_file = 'pngs/square_SPF_and_GPF_thresholds.png'
    title = 'Square Lattice'
    plot_spf_and_gpf_thresholds(in_files, styles, title, out_file)

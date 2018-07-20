# Import Python packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Adds path to data directory
sys.path.append('../analysis')

font = {'family': 'serif',
        # 'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


def plot_time_trial_data(in_file, out_file, n_min=6):
    """ Plots time-trial build time scaling data """
    plt.figure()
    fig, ax = plt.subplots()
    df = pd.read_csv(in_file, index_col=[0, 1])
    nodes = [n for n in list(df.index.levels[0]) if n >= n_min][::-1]
    colors = []
    for n in nodes:
        data = df.xs(n)
        x_vals = np.array(data.index.values)
        build_avs = np.array(data.build_t_av)
        build_std = np.array(data.build_t_std)
        mnt_avs = np.array(data.mnt_t_av)
        mnt_std = np.array(data.mnt_t_std)
        line, = ax.plot(x_vals, build_avs, linestyle='-', label=str(n))
        color = line.get_color()
        colors.append(color)
        ax.fill_between(x_vals, build_avs - 1 * build_std,
                        build_avs + 1 * build_std,
                        alpha=0.1, linewidth=0.0, color=color)
        ax.plot(x_vals, mnt_avs, linestyle='--', label=str(n), color=color)
        ax.fill_between(x_vals, mnt_avs - 1 * mnt_std, mnt_avs + 1 * mnt_std,
                        alpha=0.1, linewidth=0.0, color=color)
    node_handles = [Line2D([], [], color=color, linestyle='-',
                           linewidth=2, label=str(node))
                    for node, color in zip(nodes, colors)]
    build_line = Line2D([], [], color='k', linestyle='-', linewidth=2)
    mnt_line = Line2D([], [], color='k', linestyle='--', linewidth=2)
    line_handles = [build_line, mnt_line]
    line_labels = [r'Building $\left|\Psi\right\rangle$',
                   r'Finding $\mathcal{M}$']
    node_labels = [h.get_label() for h in node_handles]
    # handles, labels = ax.get_legend_handles_labels()
    node_lgd = plt.legend(node_handles, node_labels,
                          title=r'Nodes, $n$', loc='upper right')
    line_lgd = plt.legend(line_handles, line_labels,
                          title=r'Algorithm', loc='lower right')
    ax.add_artist(node_lgd)
    ax.add_artist(line_lgd)
    ax.set_yscale('log')
    plt.minorticks_on()
    ax.grid(True, ls='--', lw=.3, c='k', alpha=.3, which='major')
    ax.grid(True, ls=':', lw=.3, c='k', alpha=.3, which='minor')
    # plt.legend(nodes, title=r'Nodes, $|V|$', loc='lower right')
    plt.xlabel(r'Edges, $m$', size=24)
    plt.ylabel(r'Average build time, $\bar{t} \;$ (seconds)', size=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.title(title)
    ax.set_xlim(5, 95)
    ax.set_ylim(0.001)
    fig.set_size_inches(11, 7)
    plt.savefig(out_file, bbox_inches='tight')


if __name__ == '__main__':
    if not os.path.exists('pngs'):
        os.makedirs('pngs')
    in_file = 'data/ER_Gnm_time_trials_1000MC_MW2.csv'
    out_file = 'pngs/ER_Gnm_time_trials_1000MC_MW2.png'
    plot_time_trial_data(in_file, out_file)

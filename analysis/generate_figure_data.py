# Import Python modules
import os
import sys
import json
import numpy as np
# Import local modules
sys.path.append('..')
from channel_analysis import build_channel, analyse_channel
from channels.square_lattice import SquareLatticeChannel
from channels.hex_lattice import HexagonalLatticeChannel
from channels.tri_lattice import TriangularLatticeChannel
from channels.tree_to_tree import TreeToTreeChannel
from channels.crazy_graph import CrazyGraphChannel
from loss_analysis import unheralded_loss_tel_mc
from time_trials import random_gnm_build_time_trials

if __name__ == '__main__':
    verbose = False
    use_gpu = False
    workers = 1
    max_weight, rel_weights = 5, True

    # Generate data for lattice channels
    lattice_channels = [SquareLatticeChannel, HexagonalLatticeChannel,
                        TriangularLatticeChannel]
    lattice_sizes = [(2, 2), (3, 3), (4, 4)]
    mc_reps = 10000
    for channel in lattice_channels:
        for width, length in lattice_sizes:
            channel_kwargs = {'length': length, 'width': width,
                              'use_gpu': use_gpu}
            i, o = 0, width * length + 1
            prefix = 'W%dxL%d_' % (width, length) + channel.__name__
            print "Simulating the %dx%d %s" % (width, length, channel.__name__)
            build_channel(channel, channel_kwargs, o, prefix)
            analyse_channel(prefix, i, o, max_weight, rel_weights,
                            verbose=verbose, workers=workers, mc_reps=mc_reps)
            print

    # Generate Crazy Graph data
    channel = CrazyGraphChannel
    width, length = 4, 4
    channel_kwargs = {'length': length, 'width': width, 'use_gpu': use_gpu}
    i, o = 0, width * length + 1
    prefix = 'W%dxL%d_' % (width, length) + channel.__name__
    print "Simulating the 4x4 CrazyGraphChannel"
    build_channel(channel, channel_kwargs, o, prefix)
    analyse_channel(prefix, i, o, max_weight, rel_weights,
                    verbose=verbose, workers=workers, mc_reps=mc_reps)
    print

    # Generate tree-to-tree data
    channel = TreeToTreeChannel
    branches, depth = 2, 3
    channel_kwargs = {'branches': 2, 'depth': 3, 'use_gpu': use_gpu}
    max_weight, rel_weights = 5, True
    qubits = ((branches ** (depth + 1) - 1) / (branches - 1)) + \
        ((branches ** (depth) - 1) / (branches - 1)) - 1
    i, o = 0, qubits
    prefix = 'B%dxD%d_' % (branches, depth) + channel.__name__
    print "Simulating the B2xD3 TreeToTreeChannel"
    build_channel(channel, channel_kwargs, o, prefix)
    analyse_channel(prefix, i, o, max_weight, rel_weights,
                    verbose=verbose, workers=workers, mc_reps=mc_reps)
    print

    # Generates measurement strategy and maximum weight data
    print "Generating data for measurement strategy comparison"
    prefix = 'W4xL4_TriangularLatticeChannel'
    in_file = 'data/' + prefix + '_MW%d_MNT_PATS.json' \
        % (max_weight)
    mnt_pats = json.load(open(in_file, 'r'))
    mnt_pats, qubit_key = mnt_pats['mnt_pats'], mnt_pats['qubit_key']
    # Turns JSON strings back to ints
    mnt_pats = {int(key): value for key, value in mnt_pats.iteritems()}
    in_file = 'data/' + prefix + '_MW%d_SPF_LOSS_TOL_ALL.json' % (max_weight)
    all_tols = json.load(open(in_file, 'r'))
    mc_reps = 100
    loss_probs = np.linspace(0, 1, 101)
    # Simulate "max-tolerance" measurement strategy performance
    strategy = 'max_tol'
    out_file = 'data/' + prefix + \
        '_MW%d_%dMC_SPF_UH_MT_TEL_RATE.csv' % (max_weight, mc_reps)
    if not os.path.isfile(out_file):
        unheralded_loss_tel_mc(mnt_pats, all_tols, qubit_key, loss_probs,
                               mc_reps, filename=out_file, workers=workers,
                               strategy=strategy, verbose=verbose)
    # Simulate "most common" measurement strategy performance
    strategy = 'most_common'
    out_file = 'data/' + prefix + \
        '_MW%d_%dMC_SPF_UH_MC_TEL_RATE.csv' % (max_weight, mc_reps)
    if not os.path.isfile(out_file):
        unheralded_loss_tel_mc(mnt_pats, all_tols, qubit_key, loss_probs,
                               mc_reps, filename=out_file, workers=workers,
                               strategy=strategy, verbose=verbose)
    print

    # Simulates SPF performance for difference absolute maximum weights
    print "Generating data for different absolute maximum weights"
    mc_reps = 1000
    i, o = 0, 17
    rel_weights = False
    for max_weight in range(10, 15):
        analyse_channel(prefix, i, o, max_weight, rel_weights,
                        verbose=verbose, workers=workers, mc_reps=mc_reps)
    print

    print "Generating time-trial data"
    # Generates time-trial data
    n = 14
    nodes = range(6, n)
    edge_divs = 11
    mc_reps = 1000
    max_weight = 2
    filename = 'data/ER_Gnm_time_trials_%dMC_MW%d.csv' % (mc_reps, max_weight)
    random_gnm_build_time_trials(nodes, edge_divs, mc_reps, max_weight,
                                 filename=filename, workers=workers)

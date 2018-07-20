# Import Python packages
import os
import sys
from time import time
# Import local modules
sys.path.append('..')
from channels.channel import Channel
from channels.square_lattice import SquareLatticeChannel
from channels.hex_lattice import HexagonalLatticeChannel
from channels.tri_lattice import TriangularLatticeChannel
from channels.tree_to_tree import TreeToTreeChannel
from channels.crazy_graph import CrazyGraphChannel
from loss_analysis import *


def build_channel(channel, channel_kwargs, output, prefix,
                  test=False):
    """ Builds the channel and outputs data """
    # Creates data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # ================== CREATES CHANNEL AND EXPORTS TO FILE ==================
    out_file = 'data/' + prefix + '_data_ALL.json'
    if not os.path.isfile(out_file):
        print "Creating channel..."
        start = time()
        psi = channel(**channel_kwargs)
        end = time() - start
        print "Runtime = %dm %ds\n" % (end / 60, end % 60)
        print "Exporting channel to file..."
        psi.export_channel_to_file(out_file)

    # ======================== TESTS ALL STABS CORRECT ========================
    if test:
        psi._test_all_non_trivial_combos_found(print_stabs=True)
        psi._test_combo_stabs_correct()

    # ======= REMOVES STABS WITHOUT SUPPORT ON INPUT AND OUTPUT QUBITS ========
    out_file = 'data/' + prefix + '_data_JOINED.json'
    if not os.path.isfile(out_file):
        print "Removing stabilizers without support on I or O..."
        in_file = 'data/' + prefix + '_data_ALL.json'
        psi = Channel(filename=in_file)
        inputs = list(set(psi._support(psi.X_op)) |
                      set(psi._support(psi.Z_op)))
        psi.update_inputs_and_outputs(inputs=inputs, outputs=[output],
                                      join=True)
        psi.export_channel_to_file(out_file)


def analyse_channel(prefix, i, o, max_weight, rel_weights,
                    verbose=False, workers=1, mc_reps=1000):
    """ Performs a full analysis on the specified channel """

    # ======================= GETS MEASUREMENT PATTERNS =======================
    # (higher weight => deeper search for measurement patterns)
    print "Finding measurement patterns..."
    in_file = 'data/' + prefix + '_data_JOINED.json'
    out_file = 'data/' + prefix + '_MW%d_MNT_PATS.json' % (max_weight)
    if not os.path.isfile(out_file):
        psi = Channel(filename=in_file)
        mnt_pats, qubit_key = psi.get_mnt_patterns(max_weight=max_weight,
                                                   rel_weight=rel_weights)
        mnt_pats_key = {'mnt_pats': mnt_pats, 'qubit_key': qubit_key}
        with open(out_file, 'w') as fp:
            json.dump(mnt_pats_key, fp)
    else:
        mnt_pats = json.load(open(out_file, 'r'))
        mnt_pats, qubit_key = mnt_pats['mnt_pats'], mnt_pats['qubit_key']
        # Turns JSON strings back to ints
        mnt_pats = {int(key): value for key, value in mnt_pats.iteritems()}
    out_file = 'data/' + prefix + '_MW%d_LOSS_TOL_RAW.json' % (max_weight)
    if not os.path.isfile(out_file):
        loss_tol = get_loss_tolerance(mnt_pats, qubit_key)
        with open(out_file, 'w') as fp:
            json.dump(loss_tol, fp)

    # ================ IMPORTS RAW DATA AND FINDS ALL LOSS TOLS ===============
    print "Finding all loss tolerances..."
    in_file = 'data/' + prefix + '_MW%d_LOSS_TOL_RAW.json' % (max_weight)
    out_file = 'data/' + prefix + '_MW%d_SPF_LOSS_TOL_ALL.json' % (max_weight)
    if not os.path.isfile(out_file):
        loss_tols = import_loss_tols(in_file, filename=out_file)

    # ================ FINDS GRAPH PATHFINDING LOSS TOLERANCES ================
    print "Finding graph pathfinding loss tolerances..."
    in_file = 'data/' + prefix + '_data_ALL.json'
    out_file = 'data/' + prefix + '_GPF_LOSS_TOL_ALL.json'
    if not os.path.isfile(out_file):
        data = json.load(open(in_file, 'r'))
        edges = data["edges"]
        graph = nx.Graph(edges)
        graph_loss_tols(graph, i, o, filename=out_file)

    # =============== GETS PER NODE LOSS TOLERANCE FOR HEATMAPS ===============
    print "Calculating per-node loss tolerances..."
    in_file = 'data/' + prefix + '_MW%d_SPF_LOSS_TOL_ALL.json' % (max_weight)
    out_file = 'data/' + prefix + '_MW%d_SPF_PER_NODE_TOL.csv' % (max_weight)
    if not os.path.isfile(out_file):
        all_tols = json.load(open(in_file, 'r'))
        per_node_tols = get_per_node_loss_tol(all_tols, filename=out_file)

    # ============== SIMULATES SPF LOSS TOLERANCE VIA MONTE CARLO =============
    print "Simulating SPF loss tolerance..."
    in_file = 'data/' + prefix + \
        '_MW%d_SPF_LOSS_TOL_ALL.json' % (max_weight)
    out_file = 'data/' + prefix + \
        '_MW%d_%dMC_SPF_TEL_RATE.csv' % (max_weight, mc_reps)
    if not os.path.isfile(out_file):
        loss_probs = np.linspace(0, 1, 101)
        spf_loss_tols = json.load(open(in_file, 'r'))
        spf_data = heralded_loss_tel_mc(spf_loss_tols, qubit_key, loss_probs,
                                        mc_reps, filename=out_file,
                                        verbose=verbose)

    # ============== SIMULATES GPF LOSS TOLERANCE VIA MONTE CARLO =============
    print "Simulating GPF loss tolerance..."
    in_file = 'data/' + prefix + '_GPF_LOSS_TOL_ALL.json'
    out_file = 'data/' + prefix + '_%dMC_GPF_TEL_RATE.csv' % (mc_reps)
    if not os.path.isfile(out_file):
        loss_probs = np.linspace(0, 1, 101)
        gpf_loss_tols = json.load(open(in_file, 'r'))
        gpf_data = heralded_loss_tel_mc(gpf_loss_tols, qubit_key, loss_probs,
                                        mc_reps, filename=out_file,
                                        verbose=verbose)

    # ===== CALCULATES PROPORTION OF LOSS CONFIGURATIONS SPF TOLERANT TO ======
    print "Finding SPF configuration loss tolerance..."
    in_file = 'data/' + prefix + '_MW%d_SPF_LOSS_TOL_ALL.json' % (max_weight)
    out_file = 'data/' + prefix + '_MW%d_SPF_CONFIG_TOL.csv' % (max_weight)
    if not os.path.isfile(out_file):
        spf_loss_tols = json.load(open(in_file, 'r'))
        spf_data = get_qubit_no_loss_tolerances(spf_loss_tols, qubit_key,
                                                filename=out_file)

    # ===== CALCULATES PROPORTION OF LOSS CONFIGURATIONS GPF TOLERANT TO ======
    print "Finding GPF configuration loss tolerance..."
    in_file = 'data/' + prefix + '_GPF_LOSS_TOL_ALL.json'
    out_file = 'data/' + prefix + '_GPF_CONFIG_TOL.csv'
    if not os.path.isfile(out_file):
        gpf_loss_tols = json.load(open(in_file, 'r'))
        gpf_data = get_qubit_no_loss_tolerances(gpf_loss_tols, qubit_key,
                                                filename=out_file)

    # ===================== GET UNHERALDED LOSS TOLERANCE =====================
    print "Finding unheralded loss tolerances..."
    out_file = 'data/' + prefix + \
        '_MW%d_%dMC_SPF_UH_MT_TEL_RATE.csv' % (max_weight, mc_reps)
    if not os.path.isfile(out_file):
        # Loads in measurement patterns and converts JSON strings to ints
        in_file = 'data/' + prefix + '_MW%d_MNT_PATS.json' % (max_weight)
        mnt_pats = json.load(open(in_file, 'r'))
        mnt_pats, qubit_key = mnt_pats['mnt_pats'], mnt_pats['qubit_key']
        mnt_pats = {int(key): value for key, value in mnt_pats.iteritems()}
        # Loads in loss tolerances
        in_file = 'data/' + prefix + \
            '_MW%d_SPF_LOSS_TOL_ALL.json' % (max_weight)
        all_tols = json.load(open(in_file, 'r'))
        # Simulate "max-tolerance" measurement strategy performance
        strategy = 'max_tol'
        loss_probs = np.linspace(0, 1, 101)
        unheralded_loss_tel_mc(mnt_pats, all_tols, qubit_key, loss_probs,
                               mc_reps, filename=out_file, workers=1,
                               strategy=strategy, verbose=verbose)


if __name__ == '__main__':
    verbose, test, use_gpu = False, False, False
    mc_reps, workers = 1000, 1

    # =================== INPUT FOR SquareLatticeChannel ====================
    print "Initialising input variables..."
    channel = SquareLatticeChannel
    width, length = 3, 3
    channel_kwargs = {'width': width, 'length': length, 'use_gpu': use_gpu}
    max_weight, rel_weights = 5, True
    i, o = 0, width * length + 1
    prefix = 'W%dxL%d_SqLatChannel_test' % (width, length)

    # # =================== INPUT FOR HexagonalLatticeChannel =================
    # print "Initialising input variables..."
    # channel = HexagonalLatticeChannel
    # width, length = 3, 3
    # channel_kwargs = {'width': width, 'length': length}
    # max_weight, rel_weights = 5, True
    # i, o = 0, width * length + 1
    # prefix = 'W%dxL%d_HexLat' % (width, length)

    # # =============== INPUT FOR TriangularSquareLatticeChannel ==============
    # print "Initialising input variables..."
    # channel = TriangularLatticeChannel
    # width, length = 3, 3
    # channel_kwargs = {'width': width, 'length': length}
    # max_weight, rel_weights = 5, True
    # i, o = 0, width * length + 1
    # prefix = 'W%dxL%d_TriLat' % (width, length)

    # # ==================== INPUT FOR TreeToTreeChannel ======================
    # print "Initialising input variables..."
    # channel = TreeToTreeChannel
    # branches, depth = 2, 3
    # channel_kwargs = {'branches': 2, 'depth': 3}
    # max_weight, rel_weights = 5, True
    # qubits = ((branches ** (depth + 1) - 1) / (branches - 1)) + \
    #     ((branches ** (depth) - 1) / (branches - 1)) - 1
    # i, o = 0, qubits
    # prefix = 'B%dxD%d_T2T' % (branches, depth)

    # # ==================== INPUT FOR CrazyGraphChannel ======================
    # print "Initialising input variables..."
    # channel = CrazyGraphChannel
    # width, length = 3, 3
    # channel_kwargs = {'width': width, 'length': length}
    # max_weight, rel_weights = 5, True
    # i, o = 0, width * length + 1
    # prefix = 'W%dxL%d_CrazyGraph' % (width, length)

    build_channel(channel, channel_kwargs, o, prefix, test=test)
    analyse_channel(prefix, i, o, max_weight, rel_weights,
                    verbose=verbose, workers=workers, mc_reps=mc_reps)

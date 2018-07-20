# Python module
import sys
import csv
import json
import numpy as np
import networkx as nx
import itertools as it
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
from random import random, shuffle
from scipy.special import binom
from collections import Counter, defaultdict
# Import local modules
sys.path.append('..')
from utils.utils import flatten, init_worker


def get_loss_tolerance(mnt_patterns, qubit_key):
    """ Gets all possible combinations of qubit loss patterns can tolerate """
    for w, patterns in mnt_patterns.items():
        n_q = set([tuple(q for op, q in zip(pattern, qubit_key) if not op)
                   for pattern in patterns])
        mnt_patterns[w] = tuple(n_q)
    return mnt_patterns


def get_all_loss_tols(max_tols):
    """ Finds all subsets of loss tolerable by measurement patterns """
    all_tols = set()
    while max_tols:
        max_tol = max_tols.pop()
        sub_tols = set([tuple(tol) for r in range(1, len(max_tol) + 1)
                        for tol in it.combinations(max_tol, r)])
        all_tols |= sub_tols
        for sub_tol in sub_tols:
            sub_tol = list(sub_tol)
            if sub_tol in max_tols:
                max_tols.remove(sub_tol)
    all_tols = map(list, list(all_tols))
    all_tols.append([])
    return all_tols


def import_loss_tols(in_file, filename=None):
    """ Imports and formats loss tols for use. Exports to file, if provided """
    with open(in_file, 'r') as fp:
        data = json.load(fp)
    max_tols = flatten(value for value in data.values())
    all_tols = get_all_loss_tols(max_tols)
    if filename:
        with open(filename, 'w') as fp:
            json.dump(all_tols, fp)
    return all_tols


def graph_loss_tols(graph, i, o, filename=None):
    """
        Gets all configurations graph pathfinding is loss tolerant to.
        Exports output to file, if provided.
    """
    nodes = set(graph.nodes()) - set([i, o])
    all_tols = [[]]
    loss_tol_nodes = set(nodes)
    for r in range(1, len(nodes) + 1):
        loss_configs = list(it.combinations(loss_tol_nodes, r))
        loss_tol_nodes = set()
        while loss_configs:
            loss_config = loss_configs.pop()
            lost_nodes = \
                set(loss_config) | \
                set(flatten(map(graph.neighbors, loss_config)))
            loss_graph = deepcopy(graph)
            loss_graph.remove_nodes_from(lost_nodes)
            if i in loss_graph.nodes() and o in loss_graph.nodes() and \
                    nx.has_path(loss_graph, i, o):
                loss_tol_nodes |= set(loss_config)
                all_tols.append(list(loss_config))
    if filename:
        with open(filename, 'w') as fp:
            json.dump(all_tols, fp)
    return all_tols


def get_per_node_loss_tol(all_tols, filename=None):
    """
        For each qubit in the state, calculates the number of measurement
        patterns that can tolerate it's loss.
    """
    tol_counts = Counter(flatten(all_tols))
    tol_counts = [[n, count] for n, count in tol_counts.items()]
    if filename:
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['node', 'tol_count'])
            writer.writerows(tol_counts)
    return tol_counts


def most_common_mnt(avail_pats, qubit_key, measured):
    """ Picks measurement that occurs most in the available patterns """
    all_mnts = dict(Counter((q, mnt)
                            for mnt_pat in flatten(avail_pats.values())
                            for q, mnt in zip(qubit_key, mnt_pat)
                            if mnt and q not in measured))
    max_c = max(c for c in all_mnts.values())
    best_mnts = [mnt for mnt, c in all_mnts.items() if c == max_c]
    shuffle(best_mnts)
    return best_mnts.pop()


def max_tol_mnt(avail_pats, qubit_key, measured):
    """ Picks most common measurement in the most loss tolerant patterns """
    best_pats = avail_pats[min(avail_pats)]
    best_mnts = dict(Counter((q, mnt) for mnt_pat in best_pats for q, mnt
                             in zip(qubit_key, mnt_pat)
                             if mnt and q not in measured))
    max_c = max(c for c in best_mnts.values())
    best_mnts = [mnt for mnt, c in best_mnts.items() if c == max_c]
    shuffle(best_mnts)
    return best_mnts.pop()


p_funcs = {'max_tol': max_tol_mnt,
           'most_common': most_common_mnt}


def update_mnt_pats(avail_pats, q_index, basis):
    """
        Updates the set of available measurement patterns after qubit
        measurement by removing any patterns that don't contain it.
    """
    new_pats = dict()
    for w, mnt_pats in avail_pats.iteritems():
        new_mnt_pats = [pat for pat in mnt_pats if pat[q_index] == basis]
        if any(new_mnt_pats):
            new_pats.update({w: new_mnt_pats})
    return new_pats


def heralded_loss_tel_mc(loss_tols, qubit_key, loss_probs, mc_reps,
                         filename=None, verbose=False):
    """ Monte carlo simulation of teleportation with heralded loss """
    data = []
    n = len(qubit_key)
    for prob_loss in tqdm(loss_probs):
        losses = ([q for q in qubit_key[1:] if random() < prob_loss]
                  for _ in range(mc_reps))
        tels = [lost_q in loss_tols for lost_q in losses]
        tel_prob = np.mean(tels)
        tel_err = (tel_prob * (1 - tel_prob) / mc_reps) ** 0.5
        datum = [prob_loss, tel_prob, tel_err]
        if verbose:
            tqdm.write(str(datum))
        data.append(datum)
    if filename:
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['prob_loss', 'prob_tel', 'prob_tel_std'])
            writer.writerows(data)
    return data


def unheralded_loss_tel_sim(params):
    """ Simulates single instance of teleportation with unheralded loss """
    mnt_pats, n_all_tols, qubit_key, prob_loss, p_func = params
    loss = [q for q in qubit_key[1:] if random() < prob_loss]
    if loss == []:
        return True
    elif loss not in n_all_tols[len(loss)]:
        return False
    avail_pats = mnt_pats
    current_pat = [0] * len(qubit_key)
    measured = []
    while True:
        qubit, basis = p_func(avail_pats, qubit_key, measured)
        q_index = qubit_key.index(qubit)
        measured.append(qubit)
        if qubit in loss:
            basis = 0
        current_pat[q_index] = basis
        avail_pats = update_mnt_pats(avail_pats, q_index, basis)
        if not any(avail_pats):
            return False
        elif current_pat in avail_pats[min(avail_pats)]:
            return True


def unheralded_loss_tel_mc(mnt_pats, all_tols, qubit_key, loss_probs, mc_reps,
                           filename=None, workers=1, strategy='max_tol',
                           verbose=False):
    """ Monte carlo simulates teleportation with unheralded loss """
    data = []
    n_all_tols = defaultdict(list)
    for tol in all_tols:
        n_all_tols[len(tol)].append(tol)
    p_func = p_funcs[strategy]
    if workers > 1:
        # Initialises worker pool
        pool = mp.Pool(processes=workers, initializer=init_worker)
    for prob_loss in tqdm(loss_probs):
        params = mnt_pats, n_all_tols, qubit_key, prob_loss, p_func
        if workers == 1:
            tels = [unheralded_loss_tel_sim(params) for _ in range(mc_reps)]
        else:
            tels = pool.map(unheralded_loss_tel_sim, [params] * mc_reps)
        tel_prob = np.mean(tels)
        tel_err = (tel_prob * (1 - tel_prob) / mc_reps) ** 0.5
        datum = [prob_loss, tel_prob, tel_err]
        if verbose:
            tqdm.write(str(datum))
        data.append(datum)
    if filename:
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['prob_loss', 'prob_tel', 'prob_tel_std'])
            writer.writerows(data)
    return data


def get_qubit_no_loss_tolerances(loss_tols, qubit_key, filename=None):
    """ Finds tolerances of different loss configurations """
    data = []
    n = len(qubit_key)
    loss_nos = Counter(map(len, loss_tols))
    config_tol = [[q, c, binom(n, q), float(c) / binom(n, q)]
                  for q, c in loss_nos.items() if q > 0]
    if filename:
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            header = ['qubits', 'tol_configs', 'tot_configs', 'config_tol']
            writer.writerow(header)
            writer.writerows(config_tol)
    return config_tol


def get_max_weight_efficiencies(psi, max_weight, filename=None, verbose=False):
    """
        Gets loss tolerance of measurement patterns produced with different
        absolute maximum weights.
    """
    data = []
    for w in tqdm(range(1, max_weight + 1)):
        mnt_pats, qubit_key = psi.get_mnt_patterns(
            max_weight=w, rel_weight=True)
        loss_tols = get_loss_tolerance(mnt_pats, qubit_key)
        max_tols = flatten(value for value in loss_tols.values())
        all_tols = get_all_loss_tols(max_tols)
        datum = [w, len(all_tols)]
        if verbose:
            tqdm.write(str(datum))
        data.append(datum)
    if filename:
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            header = ['max_weight', 'loss_tol_configs']
            writer.writerow(header)
            writer.writerows(data)

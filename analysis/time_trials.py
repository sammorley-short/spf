# Import Python packages
import sys
import csv
import numpy as np
import itertools as it
import multiprocessing as mp
from tqdm import tqdm, trange
from time import time, sleep
# Import local modules
sys.path.append('..')
from utils.utils import enablePrint, disablePrint, init_worker
from channels.random_graphs import RandomGNMGraphChannel


def export_dataset(data, filename, header):
    """ Exports dataset to CSV file with given header """
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([header])
        writer.writerows(data)


def time_channel_simulation(params):
    """
        Times the generation of a channel with kwargs and the finding of
        measurement patterns
    """
    channel, kwargs, max_weight = params
    disablePrint()
    start_t = time()
    psi = channel(**kwargs)
    build_t = time() - start_t
    start_t = time()
    psi.get_mnt_patterns(max_weight=max_weight)
    mnt_t = time() - start_t
    enablePrint()
    return build_t, mnt_t


def channel_build_time_trial_mc(channel, kwargs, mc_reps, max_weight=0,
                                workers=1):
    """ Gets average build time for channel with given args. """
    params = channel, kwargs, max_weight
    if workers > 1:
        # Initialises worker pool
        pool = mp.Pool(processes=workers, initializer=init_worker)
        times = pool.map(time_channel_simulation, [params] * mc_reps)
        pool.close()
    else:
        times = [time_channel_simulation(params) for _ in range(mc_reps)]
    build_ts, mnt_ts = zip(*times)
    results = [np.mean(build_ts), np.std(build_ts),
               np.mean(mnt_ts), np.std(mnt_ts)]
    return results


def channel_build_time_trial_scan(channel, variables, consts, mc_reps,
                                  filename=None, workers=1):
    """ Gets the average build time for channel over vars mesh. """
    data = []
    print consts
    print variables
    var_names = variables.keys()
    var_mesh = list(it.product(*variables.values()))
    for var_set in tqdm(var_mesh):
        kwargs = dict(zip(var_names, var_set))
        kwargs.update(consts)
        build_t_av, build_t_std, mnt_t_av, mnt_t_std = \
            channel_build_time_trial_mc(channel, kwargs, mc_reps)
        datum = list(var_set) + [build_t_av, build_t_std]
        data.append(datum)
        tqdm.write(str(datum))
    if filename:
        header = var_names + \
            ['build_t_av', 'build_t_std', 'mnt_t_av', 'mnt_t_std']
        export_dataset(data, filename, header)
    return data


def random_gnm_build_time_trials(nodes, edge_divs, mc_reps, max_weight=0,
                                 filename=None, workers=1):
    """ Find build-time on Erdos-Renyi G_{n,m} graphs """
    data, simmed = [], []
    channel = RandomGNMGraphChannel
    with open(filename, 'a+') as csvfile:
        csvfile.seek(0)
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader, None)
        for n, m, bt_av, bt_std, mt_av, mt_std in reader:
            simmed.append((int(n), int(m)))
            data.append([int(n), int(m), float(bt_av), float(bt_std),
                         float(mt_av), float(mt_std)])
        writer = csv.writer(csvfile, delimiter=',')
        if not header:
            writer.writerow(['nodes', 'edges', 'build_t_av', 'build_t_std',
                             'mnt_t_av', 'mnt_t_std'])
        n_pbar = tqdm(total=len(nodes), desc='Nodes')
        for n in nodes:
            edges = np.linspace(n - 1, n * (n - 1) / 2, edge_divs, dtype=int) \
                if n * (n - 1) / 2 - n + 1 > edge_divs \
                else range(n-1, n * (n - 1) / 2 + 1)
            m_pbar = tqdm(total=len(edges), desc='Edges')
            for m in edges:
                if (n, m) not in simmed:
                    kwargs = {'nodes': n, 'n_edges': m, 'output': 1}
                    build_t_av, build_t_std, mnt_t_av, mnt_t_std = \
                        channel_build_time_trial_mc(channel, kwargs, mc_reps,
                                                    max_weight, workers)
                    datum = [n, m, build_t_av, build_t_std,
                             mnt_t_av, mnt_t_std]
                    simmed.append((n, m))
                    data.append(datum)
                    m_pbar.write(str(datum))
                    writer.writerow(datum)
                else:
                    sleep(0.05)
                m_pbar.update(1)
            m_pbar.close()
            sleep(0.05)
            n_pbar.update(1)
        n_pbar.close()
    return data


if __name__ == '__main__':
    n = 14
    nodes = range(3, n)
    edge_divs = 11
    mc_reps = 10
    max_weight = 2
    filename = 'ER_Gnm_time_trials_test_%dMC_MW%d.csv' \
        % (mc_reps, max_weight)
    workers = 3
    random_gnm_build_time_trials(nodes, edge_divs, mc_reps, max_weight,
                                 filename=filename, workers=workers)

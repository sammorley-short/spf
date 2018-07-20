# Import Python packages
import sys
import numpy as np
import itertools as it
from random import randint
from numba import cuda, autojit
# Import local modules
sys.path.append('..')
from utils import int_to_bits


@autojit
def merge_combos(source, target):
    c_s, s_s, r_s = source
    c_t, s_t, r_t = target
    if c_s & c_t == 0 and s_s & s_t > 0 and r_s & r_t == 0:
        return c_s | c_t
    else:
        return 0


merge_combos_gpu = cuda.jit(device=True)(merge_combos)


@cuda.jit
def find_all_merge_combos_kernel(sources, targets, new_combos):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, sources.shape[0], gridX):
        for y in range(startY, targets.shape[0], gridY):
            new_combos[x, y] = merge_combos_gpu(sources[x], targets[y])


def cpu_find_combo_matches(sources, targets, no_gens):
    if sources and targets:
        new_combos = np.zeros((len(sources), len(targets)), dtype=np.uint8)
        new_combos = [merge_combos(s, t)
                      for s, t in it.product(sources, targets)]
        new_combos = map(int_to_bits, set(np.unique(new_combos)) - set([0]))
        new_combos = [[0] * (no_gens-len(c)) + c for c in new_combos]
        return new_combos
    else:
        return []


def problem_chucks(sources, targets, dtype, max_bytes=10000000):
    x, y = sources.shape[0], targets.shape[0]
    o_array = np.zeros((x, y), dtype=dtype)
    all_inputs = [o_array, sources, targets]
    chunks = sum(map(sys.getsizeof, all_inputs)) / max_bytes + 1
    if chunks > 1:
        print "Splitting problem into %d chunks" % (chunks)
    return it.product([sources], np.array_split(targets, chunks))


def gpu_find_combo_matches(sources, targets, no_gens):
    if sources and targets:
        dtype = np.uint32
        sources = np.array(sources, dtype=dtype)
        targets = np.array(targets, dtype=dtype)
        new_combos = set()
        for i_sources, i_targets in problem_chucks(sources, targets, dtype):
            x, y = i_sources.shape[0], i_targets.shape[0]
            s_global_mem = cuda.to_device(i_sources)
            t_global_mem = cuda.to_device(i_targets)
            o_global_mem = cuda.device_array((x, y), dtype=dtype)
            griddim = 1
            blockdim = 512
            find_all_merge_combos_kernel[griddim, blockdim](s_global_mem,
                                                            t_global_mem,
                                                            o_global_mem)
            s_global_mem.copy_to_host()
            t_global_mem.copy_to_host()
            i_combos = o_global_mem.copy_to_host()
            new_combos |= set(np.unique(i_combos))
        new_combos = map(int_to_bits, new_combos - set([0]))
        new_combos = [[0] * (no_gens-len(c)) + c for c in new_combos]
        return new_combos
    else:
        return []

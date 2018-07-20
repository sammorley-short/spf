# Import Python packages
import os
import sys
import signal
from itertools import chain, combinations, izip


def enablePrint():
    sys.stdout = sys.__stdout__


def disablePrint():
    sys.stdout = open(os.devnull, 'w')


def init_worker():
    """ Make sure we can Ctrl-c """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def flatten(array, level=1):
    """ Flattens array to given level """
    for i in range(level):
        array = [item for sublist in array for item in sublist]
    return array


def bitwise_or(a, b):
    """ Outputs the bitsting produced by a bitwise or of a and b """
    return (i | j for i, j in izip(a, b))


def bitwise_xor(a, b):
    """ Outputs the bitsting produced by a bitwise or of a and b """
    return (i ^ j for i, j in izip(a, b))


def bitwise_and(a, b):
    """ Outputs the bitsting produced by a bitwise or of a and b """
    return (i & j for i, j in izip(a, b))


def select(l, indices):
    """ Returns list containing indexed elements of l """
    return (l[i] for i in indices)


def drop(l, indices):
    """ Returns list with indexed elements dropped """
    indices = [i for i in range(len(l)) if i not in indices]
    return list(select(l, indices))


def bits_to_int(bits):
    """ Converts list of bits into integer """
    return int(''.join(map(str, bits)), 2)


def int_to_bits(i):
    """ Converts integer into list of bits """
    return [int(x) for x in bin(i)[2:]]


def powerset(s):
    """ Returns the powerset of a list (excl. the empty set) """
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s) + 1))

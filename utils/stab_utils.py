# Import Python packages
from itertools import izip

# Define mapping from binary to Pauli operators and vice versa
binary_to_pauli = {0b00: '_', 0b01: 'X', 0b10: 'Y', 0b11: 'Z'}
pauli_to_binary = {'I': 0b00, 'X': 0b01, 'Y': 0b10, 'Z': 0b11}

# Converts binary to phase value string
binary_to_phase = {0b00: '+1', 0b01: '+i', 0b10: '-1', 0b11: '-i'}

# Applies Hadamard map, inputting single Pauli and outputting in
# (phase rotation, Pauli) form.
hadamard = {0: (0, 0), 1: (0, 3), 2: (2, 2), 3: (0, 1)}

phase_gate = {0: (0, 0), 1: (0, 2), 2: (2, 1), 3: (0, 3)}

bitflip = {0: (0, 0), 1: (0, 1), 2: (2, 2), 3: (2, 3)}

qubit_gate = {'H': hadamard, 'S': phase_gate, 'X': bitflip}

# Applies cz map, inputting single Pauli and outputting in
# (phase rotation, Pauli) form.
cz = {(0, 0): (0, 0, 0),
      (0, 1): (0, 3, 1),
      (0, 2): (0, 3, 2),
      (0, 3): (0, 0, 3),
      (1, 0): (0, 1, 3),
      (1, 1): (0, 2, 2),
      (1, 2): (2, 2, 1),
      (1, 3): (0, 1, 0),
      (2, 0): (0, 2, 3),
      (2, 1): (2, 1, 2),
      (2, 2): (0, 1, 1),
      (2, 3): (0, 2, 0),
      (3, 0): (0, 3, 0),
      (3, 1): (0, 0, 1),
      (3, 2): (0, 0, 2),
      (3, 3): (0, 3, 3)}

# Outputs phase acquired by Pauli mulitplication
pauli_multiply_phase = {(0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
                        (1, 0): 0, (1, 1): 0, (1, 2): 1, (1, 3): 3,
                        (2, 0): 0, (2, 1): 3, (2, 2): 0, (2, 3): 1,
                        (3, 0): 0, (3, 1): 1, (3, 2): 3, (3, 3): 0}


def to_pauli(stab, string=True):
    """ Converts stabilizer from binary to Pauli """
    phase, paulis = stab
    phase = binary_to_phase[phase] + " "
    paulis = [binary_to_pauli[pauli] for pauli in paulis]
    if string:
        paulis = ' '.join(paulis)
    return phase + paulis


def multiply_all_stabs(stabs):
    """ Multiplies a list of stabilizers together (can parallelize) """
    phases, ops = zip(*stabs)
    phase = sum(phases) % 4
    ops = zip(*ops)
    new_op = []
    for qubit_ops in ops:
        op = 0
        for pauli in qubit_ops:
            phase += pauli_multiply_phase[(op, pauli)]
            op = pauli ^ op
        new_op.append(op)
    return [phase % 4, new_op]


def stab_multiply(stab_a, stab_b):
    """ Multiply two stabilizers together """
    stab_a, stab_b = stab_a[:], stab_b[:]
    phases = [stab_a[0], stab_b[0]]
    paulis = zip(stab_a[1], stab_b[1])
    phase = sum(phases)
    new_stab = []
    for pauli_a, pauli_b in paulis:
        pauli_c = pauli_a ^ pauli_b
        new_stab += [pauli_c]
        phase += pauli_multiply_phase[(pauli_a, pauli_b)]
    new_stab = [phase % 4, new_stab]
    return new_stab


def is_trivial_partition(stab_a, stab_b):
    """ Returns True if stab_b is a trivial partition of stab_a """
    return all(pauli_a == pauli_b for pauli_a, pauli_b
               in izip(stab_a[1], stab_b[1]) if pauli_b) and stab_a != stab_b


def is_non_trivial_stab_pair(stab1, stab2):
    """ Checks whether there is overlap between two stabilizer combos """
    return any(a * b for a, b in izip(stab1[1], stab2[1]))


def commute(a, b):
    """ Returns True if operators commute """
    ab, ba = stab_multiply(a, b), stab_multiply(b, a)
    ab_s, ba_s = (-1) ** int(ab[0] > 2), (-1) ** int(ba[0] > 2)
    return not any(ab_s * i - ba_s * j for i, j in izip(ab[1], ba[1]))


def anticommute(a, b):
    """ Returns True if operators anticommute """
    ab, ba = stab_multiply(a, b), stab_multiply(b, a)
    ab_s, ba_s = (-1) ** int(ab[0] > 2), (-1) ** int(ba[0] > 2)
    return not any(ab_s * i + ba_s * j for i, j in izip(ab[1], ba[1]))


def weight(stab):
    """ Returns number of qubits that stabilizer acts on nontrivially """
    return len([i for i in stab[1] if i])


def trace(stab, indices):
    """ Traces out the operators acting on the given indices """
    return [stab[0], drop(stab[1], indices)]


def compatible(a, b):
    """
        Two Pauli strings are compatible if they don't contain any differing
        Pauli on a qubit.
    """
    return not any(i != j and i * j != 0 for i, j in izip(a, b))

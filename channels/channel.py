# Import Python packages
import sys
import json
import random
import decorator
import numpy as np
import itertools as it
from pprint import pprint
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
# Import local modules
sys.path.append('..')
from utils.stab_utils import *
from utils.utils import *
from utils.combo_merge import cpu_find_combo_matches, gpu_find_combo_matches


class Channel:
    """ Generic graph state representing a qubit channel """

    def __init__(self, source=0, use_gpu=False, filename=None):
        """
            Either initializes the channel as a single qubit
            or imports channel from JSON file.
        """
        if filename is not None:
            self._import_channel_from_file(filename)
        else:
            self._initialize(source, use_gpu=use_gpu)

    def __str__(self):
        """ Pretty prints the channel's state """
        # Builds vertical labelling key
        vert_labels = self._build_vert_labels()
        # Outputs logical X and Z operators and stab gens S with qubit key Q
        return "X: {}\n" \
               "Z: {}\n" \
               "S: {}\n" \
               "Q: ph {}"\
            .format(to_pauli(self.X_op),
                    to_pauli(self.Z_op),
                    ',\n   '.join(to_pauli(stab) for stab in self.stab_gens),
                    '\n      '.join(' '.join(row) for row in vert_labels))

    def _initialize(self, source, use_gpu=False):
        """
        Initializes object variables.
        """
        self.use_gpu = use_gpu
        self.qubits = [source]
        self.inputs, self.outputs = [], []
        self.X_op, self.Z_op = [0b00, [0b01]], [0b00, [0b11]]
        self.stab_gens = [[0b00, [0b00]]]
        self.combo_stabs = [[0b00, [0b00]]]
        self.gen_combos = [[1]]
        self.heralded_loss = []
        self.unheralded_loss = []

    def _import_channel_from_file(self, filename):
        """ Imports the channel's state from a previously generated JSON """
        data = json.load(open(filename, 'r'))
        for key, value in data.iteritems():
            setattr(self, key, value)

    def export_channel_to_file(self, filename):
        """ Writes channel to JSON file """
        channel_data = self.__dict__
        with open(filename, 'w') as file:
            json.dump(channel_data, file)

    def _build_vert_labels(self):
        """ Creates a vertical labelling key for easy reading of qubits """
        depth = len(str(max(self.qubits, key=lambda x: len(str(x)))))
        vert_labels = [[' ' for i in range(len(self.qubits))]
                       for j in range(depth)]
        str_labels = [str(qubit) + ' ' * (depth - len(str(qubit)))
                      for qubit in self.qubits]
        for i, label in enumerate(str_labels):
            for j, char in enumerate(label):
                vert_labels[j][i] = char
        return vert_labels

    @decorator.decorator
    def _verbose(func, *args, **kwargs):
        """ Prints the state of object before and after method's action """
        fname = func.func_name
        argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        # Prints state of object before function
        print "Before %s(%s):" % (fname, ', '.join(
            '%s=%r' % entry
            for entry in zip(argnames, args)[1:] + kwargs.items()))
        print args[0], '\n'
        # Applies function and stores any output
        result = func(*args, **kwargs)
        # Prints state of object after function
        print "\nAfter %s(%s):" % (fname, ', '.join(
            '%s=%r' % entry
            for entry in zip(argnames, args)[1:] + kwargs.items()))
        print args[0], '\n'
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        # Returns function's output
        return result

    def _test(print_stabs=True, join=False, exit=True):
        """ Runs tests after the function has been applied """
        @decorator.decorator
        def wrapper(func, *args, **kwargs):
            self = args[0]
            output = func(*args, **kwargs)
            print
            print "Testing all combo stabs correct..."
            self._test_combo_stabs_correct(exit)
            print "Testing all non-trivial combos found..."
            self._test_all_non_trivial_combos_found(print_stabs, join, exit)
            print "Tests complete"
            return output
        return wrapper

    def _print_stabs(self, stabs=None):
        """ Prints stabilizers in readable Pauli operator format """
        # Keys nodes by letters
        stabs = stabs if stabs else self.combo_stabs
        vert_labels = self._build_vert_labels()
        # Outputs logical X and Z operators and stab gens S with qubit key Q
        print "S: {}\n" \
              "Q: ph {}"\
            .format(',\n   '.join(to_pauli(stab) for stab in stabs),
                    '\n      '.join(' '.join(row) for row in vert_labels))

    def _get_indices(self, qubits):
        """ Returns a list of indices for a provided qubit list """
        return [self.qubits.index(qubit) for qubit in qubits]

    def _support(self, stab):
        """ Returns the non-trivial qubit support of a stabiliser """
        return [self.qubits[i] for i, pauli in enumerate(stab[1]) if pauli]

    def update_inputs_and_outputs(self, inputs=None, outputs=None, join=False):
        """ Updates the qubits tagged for output and removes useless combos """
        self.outputs = outputs if outputs else self.outputs
        self.inputs = list(set(self.qubits) - set(self.outputs)) \
            if not inputs else inputs
        combo_stabs = zip(self.gen_combos, self.combo_stabs)
        useful_stabs, useless_stabs = \
            self._find_useful_stab_combos(combo_stabs, join)
        for combo, stab in tqdm(useless_stabs):
            self.gen_combos.remove(combo)
            self.combo_stabs.remove(stab)

    def _find_useful_stab_combos(self, combo_stabs, join):
        """
        Finds all useful (and useless) stab combos and returns both lists.
        If join: useful => stab combos with supp on both an input AND output
        If not join: useful => all stab combos with supp on input OR output
        """
        useful_stabs, useless_stabs = [], []
        for combo, stab in tqdm(combo_stabs):
            # Checks stab combo supp overlap with inputs and outputs. If no
            # ouputs or inputs defined, ignores (i.e. sets overlap as True)
            i_supp = any(op for qubit, op in it.izip(self.qubits, stab[1])
                         if qubit in self.inputs) if self.inputs else True
            o_supp = any(op for qubit, op in it.izip(self.qubits, stab[1])
                         if qubit in self.outputs) if self.outputs else True
            # Defines usefullness condition for joined and unjoined combos
            useless = not i_supp or not o_supp if join else False
            # Appends combo and stab pair to appropriate list
            if useless and any(stab[1]):
                useless_stabs.append([combo, stab])
            else:
                useful_stabs.append([combo, stab])
        return useful_stabs, useless_stabs

    def forget_qubit(self, qubit):
        """ Removes qubit from channel memory """
        index = self.qubits.index(qubit)
        # Checks that the qubit is disentangled before forgetting it
        check_gen = [0] * len(self.qubits)
        check_gen[index] = 1
        gen_supports = [[bool(op) for op in gen[1]]
                        for gen in self.stab_gens]
        if check_gen not in gen_supports:
            raise Exception("ERROR! Qubit %s is still entangled. "
                            "You can only forget disentangled qubits."
                            % (str(qubit)))
        # If qubit is disentangled, removes support from all channel objects
        del self.qubits[index]
        bad_gen_index = gen_supports.index(check_gen)
        del self.stab_gens[bad_gen_index]
        for combo in self.gen_combos:
            del combo[bad_gen_index]
        all_ops = [self.X_op, self.Z_op] + self.stab_gens + self.combo_stabs
        for op in all_ops:
            del op[1][index]
        bad_combo = [0] * len(self.qubits)
        if bad_combo in self.gen_combos:
            bad_combo_index = self.gen_combos.index(bad_combo)
            del self.gen_combos[bad_combo_index]
            del self.combo_stabs[bad_combo_index]

    def forget_qubits(self, qubits):
        """ Forget multiple qubits """
        for qubit in qubits:
            self.forget_qubit(qubit)

    def add_qubit(self, qubit, prob_loss=0):
        """ Adds qubit to channel (if it doesn't already exist) """
        if qubit not in self.qubits:
            identity = [0b00] * len(self.qubits)
            # Adds qubit to index list and adds space to operators and combos
            self.qubits.append(qubit)
            self.X_op[1].append(0b00)
            self.Z_op[1].append(0b00)
            stabs = self.stab_gens + self.combo_stabs
            for stab in stabs:
                stab[1].append(0b00)
            for combo in self.gen_combos:
                combo.append(0)
            # Adds new stabilizer generator and relevant stab combo
            self.stab_gens.append([0, identity[:] + [0b11]])
            self.combo_stabs.append([0, identity[:] + [0b11]])
            self.gen_combos.append(identity + [1])
            if random.random() < prob_loss:
                self.unheralded_loss.append(qubit)

    def add_qubits(self, qubits, prob_loss=0):
        """ Adds multiple qubits to channel (if they don't already exist) """
        for qubit in qubits:
            self.add_qubit(qubit, prob_loss)

    def act_gate(self, qubit, gate):
        """ Applies a Hadamard (H), phase (S) or bitflip (X) gate to qubit """
        index = self.qubits.index(qubit)
        all_operators = [self.X_op, self.Z_op] + \
            self.stab_gens + self.combo_stabs
        for operator in all_operators:
            phase, operator[1][index] = qubit_gate[gate][operator[1][index]]
            operator[0] = (operator[0] + phase) % 4

    def act_gates(self, qubits_gates):
        """ Applies a sequence of gates to qubits """
        for qubit, gate in qubits_gates:
            self.act_gate(qubit, gate)

    def act_hadamard(self, qubit):
        """ Applies a hadamard to qubit """
        self.act_gate(qubit, 'H')

    def act_hadamards(self, qubits):
        """ Applies a series of hadamards to given qubits """
        for qubit in qubits:
            self.act_hadamard(qubit)

    def act_phase_gate(self, qubit):
        """ Applies a phase gate to qubit """
        self.act_gate(qubit, 'S')

    def act_phase_gates(self, qubits):
        """ Applies phase gates to qubits """
        for qubit in qubits:
            self.act_phase_gates(qubit)

    def act_bitflip(self, qubit):
        """ Applies a bitflip to qubit """
        self.act_gate(qubit, 'X')

    def act_bitflips(self, qubits):
        """ Applies bitflips to qubit """
        for qubit in qubits:
            self.act_bitflip(qubit)

    def add_node(self, node, prob_loss=0):
        """ Adds a graph state node (adds a qubit and applies a Hadamard) """
        self.add_qubit(node, prob_loss)
        self.act_hadamard(node)

    def add_nodes(self, nodes, prob_loss=0):
        """ Adds multiple graph state nodes """
        for node in nodes:
            self.add_node(node, prob_loss)

    def lose_qubit(self, qubit, heralded=True):
        """ Adds qubit to heralded or unheralded loss """
        if heralded:
            self.heralded_loss.append(qubit)
        else:
            self.unheralded_loss.append(qubit)

    def lose_qubits(self, qubits, heralded=True):
        """ Adds qubit to heralded or unheralded loss """
        if heralded:
            self.heralded_loss += qubits
        else:
            self.unheralded_loss += qubits

    def _stab_from_combo(self, combo):
        """ Calculates a stabilizer from a given generator combo """
        combo_gens = [[gen[0], gen[1][:]] for in_combo, gen
                      in zip(combo, self.stab_gens) if in_combo]
        combo_stab = reduce(stab_multiply, combo_gens)
        return combo_stab

    def _get_combo_gens(self, combo):
        """ Returns the list of gens required by combo """
        return [[gen[0], gen[1][:]] for in_combo, gen
                in zip(combo, self.stab_gens) if in_combo]

    def _combo_bitstring(self, indices):
        """ Converts a list of generator incices into a combo bitstring """
        return [int(bool(i in indices)) for i in range(len(self.stab_gens))]

    def _combo_indices(self, combo):
        """ Returns gen indices in combo """
        return [i for i, c in enumerate(combo) if c]

    def _remove_trivial_combos(self, combo_stabs=None, verbose=False,
                               test_combos=None):
        """ Removes any trivial combinations tracked """
        init_len = len(self.gen_combos)
        combo_stabs = combo_stabs if combo_stabs is not None else \
            [(i, c, s) for i, (c, s)
             in enumerate(zip(self.gen_combos, self.combo_stabs))]
        for i, combo, stab in tqdm(combo_stabs):
            non_trivial, s = \
                self._is_non_trivial_combo_internal(combo,
                                                    test_combos=test_combos)
            if stab != s:
                print "Incorrect stab found:"
                print self
                print combo
                self._print_stabs([stab, s])
                raise Exception("Incorrect stabilizer tracked")
            if not non_trivial:
                self.gen_combos.remove(combo)
                self.combo_stabs.remove(stab)
        if verbose:
            fin_len = len(self.gen_combos)
            len_diff = init_len - fin_len
            # print
            print "Initially %d stabilizer combos, now %d "\
                "(%d removed, %d tested)." % \
                (init_len, fin_len, len_diff, len(combo_stabs))

    def _support_to_combo_int(self, qubits):
        """ Converts a list of qubits to the combo integer it represents """
        return bits_to_int(self._combo_bitstring(self._get_indices(qubits)))

    @_verbose
    @_test(print_stabs=True, join=False, exit=True)
    def act_CZ(self, u, v, prob=1, checkpoints=False):
        """
        Applies a CZ between qubits u and v.
        Note: only constructive, i.e. does not un-do a CZ if already has one
        """
        _find_combo_matches = gpu_find_combo_matches if self.use_gpu \
            else cpu_find_combo_matches
        if random.random() > prob:
            return False
        # If CZ acts on source, add neighbour to self.inputs
        u_index, v_index = self.qubits.index(u), self.qubits.index(v)
        u_v_indices = (u_index, v_index)
        # Applies CZ to logical ops
        # print "Updating logical operators..."
        for log_op in [self.X_op, self.Z_op]:
            u_op, v_op = log_op[1][u_index], log_op[1][v_index]
            phase, log_op[1][u_index], log_op[1][v_index] = cz[u_op, v_op]
            log_op[0] = (log_op[0] + phase) % 4
        new_gen_indices = []
        # Applies CZ to gens and tracks updated ones
        # print "Updating generators..."
        for i, gen in enumerate(self.stab_gens):
            phase = gen[0]
            u_op, v_op = gen[1][u_index], gen[1][v_index]
            phase_shift, new_u_op, new_v_op = cz[u_op, v_op]
            new_phase = (phase + phase_shift) % 4
            if (new_phase, new_u_op, new_v_op) != (phase, u_op, v_op):
                gen[1][u_index], gen[1][v_index] = new_u_op, new_v_op
                gen[0] = new_phase
                new_gen_indices.append(i)
        # Finds all combos containing updated gens and updates them.
        # Also keeps track of stabs with new support
        # print "Updating stabilizer combos..."
        updated_combo_stabs, relevant_combos = [], []
        supp_gain_combos, supp_loss_combos, supp_hold_combos = [], [], []
        all_combo_stabs = enumerate(zip(self.gen_combos, self.combo_stabs))
        for i, (combo, stab) in all_combo_stabs:
            if any(select(combo, new_gen_indices)):
                updated_combo_stabs.append((i, combo, stab))
            old_supp = set(self._support(stab))
            u_op, v_op = stab[1][u_index], stab[1][v_index]
            phase, stab[1][u_index], stab[1][v_index] = cz[u_op, v_op]
            stab[0] = (stab[0] + phase) % 4
            new_supp = set(self._support(stab))
            supp_gain = new_supp - old_supp
            supp_loss = old_supp - new_supp
            u_v_supp = new_supp & set([u, v])
            if supp_gain:
                c_s = bits_to_int(combo)
                s_s = self._support_to_combo_int(list(supp_gain))
                r_s = self._support_to_combo_int(list(old_supp))
                supp_gain_combos.append((c_s, s_s, r_s))
            if supp_loss:
                c_s = bits_to_int(combo)
                s_s = self._support_to_combo_int(list(supp_loss))
                r_s = self._support_to_combo_int(list(new_supp))
                supp_loss_combos.append((c_s, s_s, r_s))
            if u_v_supp:
                other_supp = list(new_supp - u_v_supp)
                c_t = bits_to_int(combo)
                s_t = self._support_to_combo_int(list(u_v_supp))
                r_t = self._support_to_combo_int(other_supp)
                relevant_combos.append((c_t, s_t, r_t))
            if u_v_supp and new_supp <= old_supp:
                supp_hold_combos.append((i, combo, stab))
        # Removes any combinations that may have become trivial
        # print "Finding trivial stabilizer combos..."
        triv_combos = _find_combo_matches(supp_loss_combos, relevant_combos,
                                          len(self.stab_gens))
        triv_combo_stabs = [(combo, self.combo_stabs[i]) for i, combo
                            in enumerate(self.gen_combos)
                            if combo in triv_combos]
        # Finds the set of possible new stabilizers
        # print "Finding new stabilizer combos..."
        new_combos = _find_combo_matches(supp_gain_combos, relevant_combos,
                                         len(self.stab_gens))
        new_combos = [c for c in new_combos if c not in self.gen_combos and
                      c not in triv_combos]
        # Adds new stabilizer combinations
        added_combo_stabs = []
        # print "Adding new stabilizer combos..."
        for new_combo in tqdm(new_combos):
            new_stab = self._stab_from_combo(new_combo)
            if not any(is_trivial_partition(new_stab, stab)
                       for i, combo, stab in supp_hold_combos):
                self.gen_combos.append(new_combo)
                self.combo_stabs.append(new_stab)
                added_combo_stabs.append((len(self.gen_combos) - 1,
                                          new_combo, new_stab))
        # Removes any trivial stabilizer combinations added
        # print "Removing trivial stabilizer combos..."
        for combo, stab in triv_combo_stabs:
            self.gen_combos.remove(combo)
            self.combo_stabs.remove(stab)
        # Updates the input and output qubits
        # print "Updating inputs and outputs"
        self.update_inputs_and_outputs()
        # print "CZ complete"
        if checkpoints:
            checkpoints += '_%s-%s.json' % (str(u), str(v))
            self.export_channel_to_file(checkpoints)

    def act_CZs(self, node_pairs, prob=1, checkpoints=False):
        """ Applies multiple CZs with some probability """
        for u, v in node_pairs:
            self.act_CZ(u, v, prob=prob, checkpoints=checkpoints)

    def _detrivialise_gens(self):
        """ Searches stabs for more fundamental gens """
        # Looking for stab S_c with a in c, s.t. Q(S_c) cap Q(S_{c/a}) = 0 and
        # so S_c is new K_a and {a} -> c, c -> {a}
        combo_stabs = ((c, s) for c, s
                       in zip(self.gen_combos, self.combo_stabs)
                       if sum(c) > 1)
        gens = ((g, gen) for g, gen in enumerate(self.stab_gens))
        for (combo, stab), (g, gen) in it.product(combo_stabs, gens):
            trivial = all(gen_op == stab_op for gen_op, stab_op
                          in it.izip(gen[1], stab[1]) if stab_op)
            # If non-gen stab S_c provides half of triv bipartition and
            # contains a then replace K_a with S_c and updates other stabs
            if not trivial or not combo[g]:
                continue
            old_gens = deepcopy(self.stab_gens)
            # Replaces trivial gen with detrived one
            self.stab_gens[g] = self._stab_from_combo(combo)
            # Updates all combo stabs and notes any increased support
            supp_gains, supp_holds = {}, {}
            for i, c in enumerate(self.gen_combos):
                old_stab = self.combo_stabs[i]
                new_stab = self._stab_from_combo(c)
                self.combo_stabs[i] = new_stab
                old_supp = set(self._support(old_stab))
                new_supp = set(self._support(new_stab))
                supp_gain = new_supp - old_supp
                supp_loss = old_supp - new_supp
                if supp_gain:
                    supp_gains.update({tuple(c): supp_gain})
                elif supp_gain | supp_loss == set([]):
                    supp_holds.update({tuple(c): new_stab})
            # Iterates search for new pairs until no more found
            while True:
                good_combos = set([])
                gen_combos = [c for c in self.gen_combos if c[g]]
                # Finds any new stabs from pair of n-t stabs which now
                # share overlap that didn't before and adds
                gain_update = {}
                for gain_combo, supp_gain in supp_gains.iteritems():
                    for hold_combo, hold_stab in supp_holds.iteritems():
                        # Skips any combo pairs containing same gen
                        if any(bitwise_and(gain_combo, hold_combo)):
                            continue
                        hold_supp = set(self._support(hold_stab))
                        supp_ovlp = supp_gain & hold_supp
                        new_combo = list(bitwise_or(gain_combo,
                                                    hold_combo))
                        # Skips tracked combos or those without support overlap
                        if new_combo in self.gen_combos or not supp_ovlp:
                            continue
                        good_combos |= set([tuple(new_combo)])
                        old_gs = [old_gen for b, old_gen
                                  in zip(gain_combo, old_gens) if b]
                        old_stab = reduce(stab_multiply, old_gs)
                        old_supp = set(self._support(old_stab))
                        new_supp = set(
                            self._support(self.combo_stabs[i]))
                        supp_gain = new_supp - old_supp
                        gain_update.update({tuple(new_combo):
                                            supp_gain})
                # Tracks any newly found stabs with gained support
                supp_gain.update(gain_update)
                if good_combos == set([]):
                    break
                # Tracks any newly found combos
                for good_combo in good_combos:
                    good_stab = self._stab_from_combo(good_combo)
                    self.gen_combos.append(list(good_combo))
                    self.combo_stabs.append(good_stab)
            return False
        return True

    def _check_logical_operators_non_trivial(self):
        """
        Checks that the logical operators are not trivial and finds simplified
        form if they are.
        """
        for gen in self.combo_stabs + self.stab_gens:
            if all(i == j for i, j in it.izip(gen[1], self.X_op[1]) if i != 0):
                self.X_op = stab_multiply(gen, self.X_op)
            if all(i == j for i, j in it.izip(gen[1], self.Z_op[1]) if i != 0):
                self.Z_op = stab_multiply(gen, self.Z_op)

    @_verbose
    # @_test(print_stabs=True, join=True, exit=False)
    def pauli_measurement(self, qubit, basis, forget=True, max_detrivs=10):
        # print "Measuring qubit %s in basis %d" % (str(qubit), basis)
        # print self
        """ Makes a pauli measurement on a qubit """
        if qubit in self.unheralded_loss:
            self.unheralded_loss.remove(qubit)
            self.heralded_loss.append(qubit)
            return False
        # Gets index of qubit measured and post-measurement generator
        mnt_index = self.qubits.index(qubit)
        mnt_gen = [0, [basis * bool(i is mnt_index)
                       for i in range(len(self.qubits))]]
        # Gets all generators that anticommute with measurement
        bad_gens = [(i, gen) for i, gen in enumerate(self.stab_gens)
                    if gen[1][mnt_index] not in (0, basis)]
        if bad_gens:
            # Gets bad gen and replaces it with mnt gen
            bad_index, bad_gen = bad_gens.pop(0)
            bad_combo = self._combo_bitstring([bad_index])
            self.stab_gens[bad_index] = mnt_gen
            # Multiplies through the rest of the bad gens by the removed one
            updated_gens = [bad_index]
            for i, gen in bad_gens:
                new_gen = stab_multiply(bad_gen, gen)
                self.stab_gens[i] = new_gen
                updated_gens.append(i)
            # Multiplies through by mnt operator
            for i, gen in enumerate(self.stab_gens):
                if i != bad_index and gen[1][mnt_index]:
                    new_gen = stab_multiply(mnt_gen, gen)
                    self.stab_gens[i] = new_gen
                    updated_gens.append(i)
            # Removes all combos containing bad gen
            bad_combos_stabs = [(combo, stab) for combo, stab
                                in zip(self.gen_combos, self.combo_stabs)
                                if combo[bad_index]
                                and combo != bad_combo]
            bad_combos, bad_stabs = zip(*bad_combos_stabs) \
                if bad_combos_stabs else ([], [])
            for combo, stab in bad_combos_stabs:
                self.gen_combos.remove(combo)
                self.combo_stabs.remove(stab)
            # Updates all combos changed by mnt
            update_combos = [(i, combo) for i, combo
                             in enumerate(self.gen_combos)
                             if set(self._combo_indices(combo)) &
                             set(updated_gens)]
            for i, update_combo in update_combos:
                self.combo_stabs[i] = self._stab_from_combo(update_combo)
            # Adds potentially new combos to channel
            new_combos = [list(bitwise_xor(bad_combo, combo))
                          for combo in bad_combos] + [bad_combo]
            for new_combo in new_combos:
                if new_combo not in self.gen_combos:
                    self.gen_combos.append(new_combo)
                    self.combo_stabs.append(self._stab_from_combo(new_combo))
            if self.X_op[1][mnt_index] not in (0, basis):
                self.X_op = stab_multiply(bad_gen, self.X_op)
            if self.Z_op[1][mnt_index] not in (0, basis):
                self.Z_op = stab_multiply(bad_gen, self.Z_op)
        else:
            # If mnt gen is already in gens, gets index, else picks other gen
            # that contains mnt_basis
            triv_i, detrived_i = None, []
            for i, gen in enumerate(self.stab_gens):
                if gen[1] == mnt_gen[1]:
                    triv_i = i
                elif gen[1][mnt_index] == basis:
                    detrived_i.append(i)
            if triv_i is None:
                triv_i = detrived_i.pop()
            # print triv_i, detrived_i
            self.stab_gens[triv_i] = mnt_gen
            # Multiplies the rest through by the mnt gen
            for i in detrived_i:
                self.stab_gens[i] = stab_multiply(mnt_gen, self.stab_gens[i])
            # Removes all combos containing the mnt gen
            bad_combos_stabs = ((c, s) for c, s
                                in zip(self.gen_combos, self.combo_stabs)
                                if c[triv_i])
            for c, s in bad_combos_stabs:
                self.gen_combos.remove(c)
                self.combo_stabs.remove(s)
            mnt_combo = [int(i == triv_i) for i in range(len(self.qubits))]
            self.gen_combos.append(mnt_combo)
            self.combo_stabs.append(mnt_gen)
            # Updates all stabs containing de-trived gens
            update_combo_stabs = ((i, c) for i, c
                                  in enumerate(self.gen_combos)
                                  if any(select(c, detrived_i)))
            for i, c in update_combo_stabs:
                self.combo_stabs[i] = self._stab_from_combo(c)
        # print "After measurement update:"
        # print self
        # print "Detriviaising generators"
        detrived, detriv_attempts = False, 0
        while not detrived and detriv_attempts < max_detrivs:
            detriv_attempts += 1
            detrived = self._detrivialise_gens()
            # print "After detrivialising generators:"
            # print self
        # Removes any trivial combos
        self._remove_trivial_combos()
        self._check_logical_operators_non_trivial()
        if forget:
            self.forget_qubit(qubit)
        # Checks all gens non-triv and removes any erroneous stabs and combos
        self.update_inputs_and_outputs()
        return True

    def pauli_measurements(self, mnts, forget=True, max_detrivs=10):
        """ Performs a sequence of Pauli measurements """
        for qubit, basis in mnts:
            self.pauli_measurement(qubit, basis, forget, max_detrivs)

    def _is_non_trivial_combo_exhuastive(self, combo, return_test_stab=False):
        """ Exhaustively tests for stabilizer triviality """
        combo_gens = [[gen[0], gen[1][:]] for in_combo, gen
                      in zip(combo, self.stab_gens) if in_combo]
        combo_stab = reduce(stab_multiply, combo_gens)
        if sum(combo) == 1:
            return True, combo_stab
        # Gets max c for (len(qubits) choose c)-fold combinations to test
        c = sum(combo) / 2
        # Tries iteratively larger i-fold combinations of gens
        for i in range(1, c + 1):
            # Tests each of the i-fold combinations for triviality
            for test_gens in it.combinations(combo_gens, i):
                test_stab = reduce(stab_multiply, test_gens)
                # Asks if each operator matches that on the same qubit in combo
                trace_match = [combo_stab[1][j] == op
                               for j, op in enumerate(test_stab[1]) if op]
                # If all match then combination is trivial
                if trace_match and all(trace_match):
                    if return_test_stab:
                        return False, test_stab
                    else:
                        return False, combo_stab
        return True, combo_stab

    def _is_non_trivial_combo_internal(self, combo, return_test_stab=False,
                                       test_combos=None):
        """ Checks stabilizer triviality by testing with internal stabs """
        combo_stab = self._stab_from_combo(combo)
        if sum(combo) == 1:
            return True, combo_stab
        gen_indices = self._combo_indices(combo)
        test_combos = test_combos if test_combos else \
            (self._combo_bitstring(indices) for indices
             in powerset(gen_indices))
        for test_combo in test_combos:
            if test_combo in self.gen_combos and test_combo != combo:
                test_stab = self.combo_stabs[self.gen_combos.index(test_combo)]
                if any(test_stab[1]) and \
                    not any(pauli_a != pauli_b for pauli_a, pauli_b
                            in it.izip(combo_stab[1], test_stab[1])
                            if pauli_b):
                    if return_test_stab:
                        return False, test_stab
                    else:
                        return False, combo_stab
        return True, combo_stab

    def _get_sorted_log_ops(self, log_op):
        """
            Gets a dictionary of logical operators sorted by their output
            qubit operator and weight
        """
        output_indices = self._get_indices(self.outputs)
        input_indices = self._get_indices(self.inputs)
        spanning_stabs = [stab for stab in self.combo_stabs
                          if any(select(stab[1], input_indices))
                          and any(select(stab[1], output_indices))]
        log_ops = [stab_multiply(log_op, stab) for stab in spanning_stabs
                   if is_non_trivial_stab_pair(log_op, stab)]
        sorted_log_ops = {b: defaultdict(list) for b
                          in it.product((1, 2, 3), repeat=len(output_indices))}
        for log_op in log_ops:
            out_op = tuple(select(log_op[1], output_indices))
            sorted_log_ops[out_op][weight(log_op)].append(log_op)
        return sorted_log_ops

    def get_mnt_patterns(self, max_weight=None, rel_weight=True):
        """
        Gets all measurement patterns made from pairs of stabilizers up
        to some maximum weight. (Here max weight is the absolute maximum
        weight if rel_weight is False, otherwise it is the max weight above the
        lowest-weight logical op for each logical operator class).
        """
        max_weight = max_weight if max_weight else len(self.qubits)
        output_indices = self._get_indices(self.outputs)
        loss_indices = self._get_indices(self.heralded_loss)
        # Finds all pairs of anticommuting logical operators
        # print "Calculating all logical operators"
        x_ops = self._get_sorted_log_ops(self.X_op)
        z_ops = self._get_sorted_log_ops(self.Z_op)
        op_pairings = [(i, j) for i, j in it.product(x_ops, z_ops)
                       if not commute([0, list(i)], [0, list(j)])]
        # Checks log op pairs for compatibility in order of weight and finds
        # any valid measurement patterns.
        mnt_patterns = defaultdict(list)
        for x_out, z_out in op_pairings:
            if rel_weight:
                x_weights = sorted(x_ops[x_out])[:max_weight]
                z_weights = sorted(z_ops[z_out])[:max_weight]
            else:
                x_weights = [x_w for x_w in x_ops[x_out] if x_w <= max_weight]
                z_weights = [z_w for z_w in z_ops[z_out] if z_w <= max_weight]
            for x_w, z_w in it.product(x_weights, z_weights):
                op_pairs = it.product(x_ops[x_out][x_w], z_ops[z_out][z_w])
                for x_op, z_op in op_pairs:
                    if compatible(drop(x_op[1], output_indices),
                                  drop(z_op[1], output_indices)) \
                            and not any(select(x_op[1], loss_indices)) \
                            and not any(select(z_op[1], loss_indices)):
                        mnt_pat = drop([max([i, j]) for i, j
                                        in zip(x_op[1], z_op[1])],
                                       output_indices + loss_indices)
                        mnt_weight = len(filter(None, mnt_pat))
                        mnt_patterns[mnt_weight].append(mnt_pat)
        # Gets measurement patterns' qubit key
        ignore_qubits = self.outputs + self.heralded_loss
        qubit_key = [qubit for qubit in self.qubits
                     if qubit not in ignore_qubits]
        return dict(mnt_patterns), qubit_key

    def find_parity_checks(self, mnt_pat, qubit_key):
        """
            Finds all possible single-measurement parity checks for a given
            measurement pattern.
        """
        q_indices = self._get_indices(qubit_key)
        parity_checks = [list(select(stab[1], qubit_key))
                         for stab in self.combo_stabs
                         if compatible(stab[1], mnt_pat + [0])]
        parity_mnts = defaultdict(list)
        for parity_check in parity_checks:
            mnts = [p * int(p == m) for p, m in zip(parity_check, mnt_pat)]
            check = [p - m for p, m in zip(parity_check, mnts)]
            if len(mnts) - mnts.count(0) == 1:
                mnt = [(qubit_key[e], mnt)
                       for e, mnt in enumerate(mnts) if mnt][0]
                parity_mnts[tuple(mnt)] += [check]
        return dict(parity_mnts)

    def _no_duplicated_combos(self, exit=True):
        """ Ensures there are no duplicated combos """
        combos = map(tuple, self.gen_combos)
        if len(set(combos)) != len(combos):
            duplicates = [combo for combo in combos if combos.count(combo) > 1]
            print "Duplicated combinations"
            pprint(set(duplicates))
            if exit:
                raise Exception('Duplicate stabilizer combinations found.')
            return False
        return True

    def _test_combo_stabs_correct(self, exit=True):
        """ Tests that the combo stabs are correct and non-trivial """
        passed = True
        for combo, stab in zip(self.gen_combos, self.combo_stabs):
            non_trivial, target = self._is_non_trivial_combo_exhuastive(combo)
            # Checks whether combo trivial
            if not non_trivial:
                print "Trivial combo"
                print combo
                self._print_stabs([stab, self._stab_from_combo(combo)])
                non_trivial, stab = self._is_non_trivial_combo_exhuastive(
                    combo, True)
                self._print_stabs([stab])
                print self
                passed = False
            # Checks whether stab combo matches that tracked in channel
            if stab != target:
                print "Incorrect stabilizer (correct stab on bottom)"
                print combo
                self._print_stabs([stab, target])
                print
                passed = False
            if not non_trivial and exit:
                raise Exception('Trivial stabilizer found.')
            if stab != target and exit:
                raise Exception('Incorrect stabilizer found.')
        no_dups = self._no_duplicated_combos(exit)
        return passed and no_dups

    def _test_all_non_trivial_combos_found(self, print_stabs=False,
                                           join=False, exit=True,
                                           verbose=True):
        """
            Tests all non-trivial stabilizers have been found and checks no
            trivial stabilizers are tracked.
        """
        # Gets indices of any all-identity generators
        null_gen_indices = [e for e, gen in enumerate(self.stab_gens)
                            if sum(gen[1]) == 0]
        # Gets all non-trivial stabilizers
        all_combos = list(it.product((0, 1), repeat=len(self.qubits)))[1:]
        nt_combos = [(combo,) + self._is_non_trivial_combo_exhuastive(combo)
                     for combo in all_combos]
        nt_combos = [[combo, stab] for combo, non_trivial, stab
                     in nt_combos if non_trivial]
        # If only joined stabs tracked, removes any without support on I and O
        if join:
            nt_combos, useless_stabs = \
                self._find_useful_stab_combos(nt_combos, join)
        nt_combos = [combo for combo, stab in nt_combos]
        # Removes combos containing null gen (excl. null combo)
        nt_combos = [combo for combo in nt_combos
                     if not any(select(combo, null_gen_indices))]
        nt_combos += [[int(i == j) for i in range(len(self.stab_gens))]
                      for j in null_gen_indices]
        if verbose:
            print "%d non-trivial combos possible" % (len(nt_combos))
            print "%d non-trivial combos tracked" % (len(self.gen_combos))
        # Displays any tracked or untracked combos found
        if len(nt_combos) != len(self.gen_combos):
            untracked_combos = \
                set(map(tuple, nt_combos)) - set(map(tuple, self.gen_combos))
            overtracked_combos = \
                set(map(tuple, self.gen_combos)) - set(map(tuple, nt_combos))
            if untracked_combos and print_stabs:
                print "UNTRACKED COMBOS (%d):" % (len(untracked_combos))
                for combo in untracked_combos:
                    combo = tuple(combo)
                    print combo, [e for e, i in enumerate(combo) if i]
                    untracked_gens = self._get_combo_gens(combo)
                    untracked_stab = reduce(stab_multiply, untracked_gens)
                    self._print_stabs([untracked_stab] + untracked_gens)
            if overtracked_combos and print_stabs:
                print "OVERTRACKED COMBOS (%d):" % (len(overtracked_combos))
                for combo in overtracked_combos:
                    combo = tuple(combo)
                    print combo, [e for e, i in enumerate(combo) if i]
                    overtracked_gens = self._get_combo_gens(combo)
                    overtracked_stab = reduce(stab_multiply, overtracked_gens)
                    self._print_stabs([overtracked_stab] + overtracked_gens)
                    non_trivial, stab = self._is_non_trivial_combo_exhuastive(
                        combo, True)
                    self._print_stabs([stab])
            # Raises exception if desired
            if exit:
                raise Exception('Untracked and/or overtracked combos found.')
            return False
        else:
            return True


if __name__ == '__main__':
    # ============================ 3-WAYS EXAMPLE =============================
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10),
             (2, 5), (5, 8), (8, 10),  (3, 6), (6, 9), (9, 10)]
    psi = Channel(0)
    psi.add_nodes(nodes)
    psi.act_CZs(edges)
    psi.update_inputs_and_outputs(inputs=[0, 1, 2, 3],
                                  outputs=[10], join=True)

    mnt_pats, qubit_index = psi.get_mnt_patterns()
    print sum([len(pats) for pats in mnt_pats.values()])

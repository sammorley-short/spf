# Import Python packages
import sys
import random
import string
import unittest
from tqdm import tqdm
from pprint import pprint
# Import local modules
sys.path.append('..')
from channels.random_graphs import RandomGNPGraphChannel, GraphChannel
from utils.stab_utils import anticommute
from utils.utils import enablePrint, disablePrint, flatten


class TestChannel(unittest.TestCase):

    def setUp(self):
        pass

    def test_RandomGNPGraphChannel_gpu(self):
        """ Tests the construction of random G(n,p) graphs using the GPU """
        nodes = 8
        for i in tqdm(range(100)):
            prob_edge = random.uniform(0.2, 0.5)
            disablePrint()
            psi = RandomGNPGraphChannel(nodes, prob_edge, use_gpu=True)
            enablePrint()
            all_nt = psi._test_all_non_trivial_combos_found(verbose=False)
            stab_combos_correct = psi._test_combo_stabs_correct()
            self.assertTrue(all_nt)
            self.assertTrue(stab_combos_correct)

    def test_RandomGNPGraphChannel_cpu(self):
        """ Tests the construction of random G(n,p) graphs using the CPU """
        nodes = 8
        for i in tqdm(range(100)):
            prob_edge = random.uniform(0.2, 0.5)
            disablePrint()
            psi = RandomGNPGraphChannel(nodes, prob_edge, use_gpu=False)
            enablePrint()
            all_nt = psi._test_all_non_trivial_combos_found(verbose=False)
            stab_combos_correct = psi._test_combo_stabs_correct()
            self.assertTrue(all_nt)
            self.assertTrue(stab_combos_correct)

    def test_qubit_names_unimportant(self):
        """ Makes sure nodes can be strings as well as integers """
        n = 8
        nodes = [0] + list(string.ascii_lowercase)[:(n-1)]
        for i in tqdm(range(100)):
            prob_edge = random.uniform(0.2, 0.5)
            disablePrint()
            psi = RandomGNPGraphChannel(nodes, prob_edge)
            enablePrint()
            all_nt = psi._test_all_non_trivial_combos_found(verbose=False)
            stab_combos_correct = psi._test_combo_stabs_correct()
            self.assertTrue(all_nt)
            self.assertTrue(stab_combos_correct)

    def test_measurement_patterns_teleport(self):
        """ Finds measurement patterns and tests they teleport the state """
        nodes = 8
        output = nodes - 1
        for i in tqdm(range(100)):
            prob_edge = random.uniform(0.2, 0.5) / 2
            disablePrint()
            psi = RandomGNPGraphChannel(nodes, prob_edge, output=output,
                                        use_gpu=True)
            psi.update_inputs_and_outputs()
            enablePrint()
            mnt_patterns, qubit_key = psi.get_mnt_patterns()
            mnt_pattern = random.choice(flatten(mnt_patterns.values()))
            mnt_pattern = [(qubit, basis) for qubit, basis in
                           zip(qubit_key, mnt_pattern) if basis]
            random.shuffle(mnt_pattern)
            for qubit, basis in mnt_pattern:
                try:
                    psi.pauli_measurement(qubit, basis, forget=False)
                except Exception:
                    print "Measurement failed"
                    print psi.edges()
                    print mnt_pattern
                    sys.exit()
                try:
                    all_nt = psi._test_all_non_trivial_combos_found(
                        print_stabs=True, join=False, verbose=False)
                except Exception:
                    print "Measurement failed"
                    print psi.edges()
                    print mnt_pattern
                    pprint(psi.gen_combos)
                    psi._print_stabs()
                    sys.exit()
                stab_combos_correct = psi._test_combo_stabs_correct()
                self.assertTrue(all_nt)
                self.assertTrue(stab_combos_correct)
            self.assertEqual(psi._support(psi.X_op), [output])
            self.assertEqual(psi._support(psi.Z_op), [output])
            self.assertTrue(anticommute(psi.X_op, psi.Z_op))


def single_shot_teleportation(nodes, edges, mnt_pattern):
    """
        Test specific graph state and measurement pattern for debugging
        purposes. Assumes input is first node and output is last in nodes list.
    """
    psi = GraphChannel(nodes, edges, output=nodes[-1])
    psi._test_combo_stabs_correct()
    psi._test_all_non_trivial_combos_found(print_stabs=True, verbose=True)
    print psi
    for qubit, basis in mnt_pattern:
        print qubit, basis
        psi.pauli_measurement(qubit, basis, forget=False)
        print psi
        psi._test_combo_stabs_correct()
        psi._test_all_non_trivial_combos_found(print_stabs=True, join=False,
                                               verbose=True)


if __name__ == '__main__':
    unittest.main()

    # n = 8
    # nodes = range(n)
    # edges = [(0, 3), (0, 5), (1, 4), (1, 6), (2, 3), (2, 7), (4, 5)]
    # mnt_pattern = [(2, 1), (0, 1), (4, 3), (5, 2), (3, 1)]
    # single_shot_teleportation(nodes, edges, mnt_pattern)

# Import local modules
from channel import Channel


class TriangularLatticeChannel(Channel):
    """
        A channel produced by a 2D Triangular lattice cluster state with the
        diagonals alternating or not
    """

    def __init__(self, width, length, prob_edge=1, prob_loss=0,
                 alt_diag=True, output_node=True, checkpoints=None,
                 use_gpu=False):
        """ Initializes the lattice """
        self._initialize(0, use_gpu=use_gpu)
        self.act_hadamard(0)
        self.width = width
        self.near_layer = 0
        self.far_layer = 1
        self.prob_edge = prob_edge
        self.prob_loss = prob_loss
        self.checkpoints = checkpoints
        nodes = range(1, self.width + 1)
        v_edges = [(0, i) for i in nodes]
        self.edges = v_edges
        self.add_nodes(nodes, prob_loss)
        self.act_CZs(v_edges, checkpoints=self.checkpoints)
        r_edges = [(i, i + 1) for i in nodes[:-1]]
        self.edges += r_edges
        self.act_CZs(r_edges, prob=prob_edge, checkpoints=self.checkpoints)
        for l in range(2, length + 1):
            self.add_layer(l, alt_diag)
        if output_node:
            self.add_output_node()
        self.update_inputs_and_outputs()

    def add_layer(self, layer, alt_diag):
        """ Adds another layer to the channel """
        new_nodes = [i + self.far_layer * self.width
                     for i in range(1, self.width + 1)]
        old_nodes = [i - self.width for i in new_nodes]
        if self.far_layer > 2:
            old_nodes += [i - 2 * self.width for i in new_nodes]
        new_edges = [(i, i + 1) for i in new_nodes[:-1]]
        new_edges += zip(new_nodes, old_nodes)
        if int(alt_diag) * (layer % 2):
            new_edges += zip(new_nodes[1:], old_nodes[:-1])
        else:
            new_edges += zip(new_nodes[:-1], old_nodes[1:])
        self.edges += new_edges
        self.add_nodes(new_nodes, self.prob_loss)
        self.update_inputs_and_outputs(outputs=new_nodes + old_nodes)
        self.act_CZs(new_edges, prob=self.prob_edge,
                     checkpoints=self.checkpoints)
        self.far_layer += 1

    def add_output_node(self):
        """ Adds output node to far layer of qubits """
        far_node = max(self.qubits) + 1
        self.add_node(far_node)
        far_nodes = [i + (self.far_layer - 1) * self.width
                     for i in range(1, self.width + 1)]
        new_edges = [(far_node, i) for i in far_nodes]
        self.edges += new_edges
        self.act_CZs(new_edges, checkpoints=self.checkpoints)
        self.update_inputs_and_outputs(outputs=[far_node], join=False)

    def deconstruct(self):
        self.act_CZs(self.edges)


if __name__ == '__main__':
    # ============ SINGLE-SHOT TESTING OF TriangularLatticeChannel ============
    width = 3
    init_length = 3
    prob_edge = 1
    psi = TriangularLatticeChannel(width, init_length, prob_edge)
    psi._test_all_non_trivial_combos_found(print_stabs=True)
    psi._test_combo_stabs_correct()
    mnt_pats, qubit_index = psi.get_mnt_patterns()
    print sum([len(pats) for pats in mnt_pats.values()])

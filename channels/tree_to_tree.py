# Import local modules
from channel import Channel


class TreeToTreeChannel(Channel):
    """ A channel made of two trees, co-joined at the leaves """

    def __init__(self, branches, depth, prob_loss=0, checkpoints=None,
                 use_gpu=False):
        self._initialize(0, use_gpu=use_gpu)
        self.act_hadamard(0)
        self.prob_loss = prob_loss
        self.checkpoints = checkpoints
        self.edges = []
        prev_layer = [0]
        for l in range(1, depth + 1):
            next_layer = []
            for i, node in enumerate(prev_layer):
                neighs = [j + prev_layer[-1] + i * branches
                          for j in range(1, branches + 1)]
                edges = [(node, neigh) for neigh in neighs]
                self.add_nodes(neighs)
                self.act_CZs(edges, checkpoints=self.checkpoints)
                self.edges += edges
                next_layer += neighs
            prev_layer = next_layer
        for l in range(1, depth + 1):
            next_layer = []
            for i in range(len(prev_layer) / branches):
                nodes = prev_layer[i * branches:i * branches + branches]
                neigh = prev_layer[-1] + 1 + i
                edges = [(node, neigh) for node in nodes]
                self.add_node(neigh)
                self.act_CZs(edges, checkpoints=self.checkpoints)
                self.edges += edges
                next_layer += [neigh]
            prev_layer = next_layer
        self.update_inputs_and_outputs(outputs=prev_layer, join=False)


if __name__ == '__main__':
    # =============== SINGLE-SHOT TESTING OF TreeToTreeChannel ===============
    depth = 2
    branches = 2
    psi = TreeToTreeChannel(branches, depth)
    psi._test_all_non_trivial_combos_found(print_stabs=True)
    psi._test_combo_stabs_correct()
    mnt_pats, qubit_index = psi.get_mnt_patterns()
    print sum([len(pats) for pats in mnt_pats.values()])

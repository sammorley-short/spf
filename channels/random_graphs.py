# Import Python packages
import networkx as nx
from random import random
# Import local modules
from channel import Channel


class GraphChannel(Channel):
    """ A channel defined by some edge set """

    def __init__(self, nodes, edges, prob_loss=0, output=None, use_gpu=False):
        """ Initialises the graph state """
        self.prob_loss = prob_loss
        self.edges = edges
        self._initialize(nodes[0], use_gpu=use_gpu)
        self.act_hadamard(nodes[0])
        self.add_nodes(nodes[1:], prob_loss)
        self.act_CZs(edges)
        self.update_inputs_and_outputs(outputs=[output])


def random_connected_graph_edges(nxg_gen, nodes, arg, output=None):
    """ Generates a Erdos-Renyi G_{n,p} random graph """
    n = len(nodes)
    relabel = {i: node for node, i in zip(nodes, range(n))}
    g = nx.Graph([(0, 1), (2, 3)])
    while not nx.is_connected(g):
        g = nxg_gen(n, arg)
        if output and g.has_edge(0, output):
            g.remove_edge(0, output)
    g = nx.relabel_nodes(g, relabel)
    return g.edges()


class RandomGNPGraphChannel(Channel):
    """ A channel produced by a G_{n,p} Erdos-Renyi random graph """

    def __init__(self, nodes, prob_edge, prob_loss=0, output=None,
                 use_gpu=False):
        """ Initialises the random graph state """
        self.prob_loss = prob_loss
        self.prob_edge = prob_edge
        nodes = range(nodes) if type(nodes) == int else nodes
        nxg_gen = nx.fast_gnp_random_graph
        edges = random_connected_graph_edges(nxg_gen, nodes, prob_edge, output)
        self.edges = edges
        self._initialize(nodes[0], use_gpu=use_gpu)
        self.act_hadamard(nodes[0])
        self.add_nodes(nodes[1:], prob_loss)
        self.act_CZs(edges)
        self.update_inputs_and_outputs(outputs=[output])


class RandomGNMGraphChannel(Channel):
    """ A channel produced by a G_{n,m} Erdos-Renyi random graph """

    def __init__(self, nodes, n_edges, prob_loss=0, output=None,
                 use_gpu=False):
        """ Initialises the random graph state """
        self.prob_loss = prob_loss
        self.n_edges = n_edges
        nodes = range(nodes) if type(nodes) == int else nodes
        n = len(nodes)
        if n_edges < n - 1 or n_edges > n * (n - 1) / 2:
            raise Exception("|E| must be such in the interval "
                            "|V| - 1 <= |E| <= |V|(|V| - 1)/2")
        nxg_gen = nx.dense_gnm_random_graph
        edges = random_connected_graph_edges(nxg_gen, nodes, n_edges, output)
        self.edges = edges
        self._initialize(nodes[0], use_gpu=use_gpu)
        self.act_hadamard(nodes[0])
        self.add_nodes(nodes[1:], prob_loss)
        self.act_CZs(edges)
        self.update_inputs_and_outputs(outputs=[output])


if __name__ == '__main__':
    # =================== TESTING RandomGNPGraphChannel ======================
    nodes = 14
    for i in range(1000):
        prob_edge = random() / 3
        print "Prob Edge = ", prob_edge
        psi = RandomGNPGraphChannel(nodes, prob_edge)
        print psi
        print "Testing channel"
        psi._test_all_non_trivial_combos_found(print_stabs=True)
        psi._test_combo_stabs_correct()

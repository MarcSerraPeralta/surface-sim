"""Module that implement some utility functions."""

from collections import deque
from itertools import count

import networkx as nx

from .layout import Layout


def set_coords(layout: Layout, override: bool = False) -> None:
    """
    set_coords Sets the coordinates of the nodes in the layout.

    Parameters
    ----------
    layout : Layout
        The layout to set the coordinates of.
    """
    if (not override) and (nx.get_node_attributes(layout.graph, "coords") != {}):
        raise ValueError("'layout' already has coordinates, use 'override' flag.")

    # Get the shift in the coordinate for a given direction.
    def get_shift(direction: str) -> int:
        if direction in ("south", "west"):
            return -1
        return 1

    nodes = list(layout.graph.nodes)  # graph nodes
    init_node = nodes.pop()  # initial node
    init_coord = [0, 0]  # initial coordinates

    set_nodes = set()  # Nodes we have already set the coordinates of.

    queue = deque()  # Queue of nodes to set the coordinates of.

    queue.appendleft((init_node, init_coord))
    while queue:
        node, coords = queue.pop()

        layout.graph.nodes[node]["coords"] = coords
        set_nodes.add(node)

        for _, nbr_node, ord_dir in layout.graph.edges(node, data="direction"):
            if nbr_node not in set_nodes:
                card_dirs = ord_dir.split("_")
                shifts = tuple(map(get_shift, card_dirs))
                nbr_coords = list(map(sum, zip(coords, shifts)))
                queue.appendleft((nbr_node, nbr_coords))


def index_chain(layout: Layout, init_node: str) -> None:
    """
    index_chain Indexes the chain of qubits in the layout.

    Parameters
    ----------
    layout : Layout
        The layout to index the chain of.
    init_node : str
        The initial qubit of the chain.

    Raises
    ------
    ValueError
        If the initial qubit is not in the graph.
    ValueError
        If any qubit is not connected to any other qubits.
    ValueError
        If any qubit is connected to more than 2 other qubits.
    """
    nodes = list(layout.graph.nodes)
    chain_inds = count(0, 1)

    if init_node not in nodes:
        raise ValueError("init_node not in graph")

    set_nodes = set()
    queue = deque()

    queue.appendleft(init_node)

    while queue:
        node = queue.pop()
        ind = next(chain_inds)
        layout.graph.nodes[node]["chain_ind"] = ind
        set_nodes.add(node)

        neighbors = list(layout.graph.adj[node])
        num_neighbors = len(neighbors)
        if num_neighbors > 2:
            raise ValueError(
                f"Qubit {node} is connected to {num_neighbors} other qubits, expected at most 2."
            )
        if num_neighbors == 0:
            raise ValueError(
                f"Qubit {node} is not connected to any other qubits, expected at least 1."
            )

        for nbr_node in neighbors:
            if nbr_node not in set_nodes:
                queue.appendleft(nbr_node)

    for node in nodes:
        if node not in set_nodes:
            layout.graph.nodes[node]["chain_ind"] = None

import pytest
import networkx as nx

from surface_sim.layouts.operations import check_overlap_layouts
from surface_sim.layouts import rot_surface_code


def test_check_non_overlapping_layouts():
    layout_1 = rot_surface_code(1)
    layout_2 = rot_surface_code(2)
    layout_2._log_qubits = ["L1"]
    layout_2._qubit_inds = {"D2": 10}
    graph = nx.DiGraph()
    graph.add_node("D2", coords=(11, 11))
    layout_2.graph = graph

    check_overlap_layouts(layout_1, layout_2)

    with pytest.raises(ValueError):
        check_overlap_layouts(layout_1, layout_1)

    return

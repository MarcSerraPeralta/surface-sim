from copy import deepcopy

import pytest

from surface_sim.layouts import rot_surface_code
from surface_sim.layouts.operations import (
    check_code_definition,
    check_overlap_layouts,
    overwrite_interaction_order,
)


def test_check_non_overlapping_layouts():
    layout_1 = rot_surface_code(
        1,
        init_ind=10,
        init_data_qubit_id=100,
        init_point=(11, 11),
        init_logical_ind=2,
        logical_qubit_label="L1",
    )
    layout_2 = rot_surface_code(2)

    check_overlap_layouts(layout_1, layout_2)

    with pytest.raises(ValueError):
        check_overlap_layouts(layout_1, layout_1)

    return


def test_check_code_definition():
    layout = rot_surface_code(3)

    check_code_definition(layout)

    # make a bad layout
    assert layout._log_qubits == {
        "L0": {"log_x": ["D1", "D4", "D7"], "log_z": ["D1", "D2", "D3"], "ind": 0}
    }
    layout._log_qubits["L0"]["log_x"] = ["D1", "D2", "D7"]

    with pytest.raises(ValueError):
        check_code_definition(layout)

    return


def test_overwrite_interaction_order():
    layout = rot_surface_code(3)
    previous_interaction_order = deepcopy(layout.interaction_order)
    new_schedule = {}
    for anc in layout.anc_qubits:
        support = list(layout.get_neighbors([anc]))
        support += [None] * (4 - len(support))
        new_schedule[anc] = support

    overwrite_interaction_order(layout, new_schedule)

    assert layout.interaction_order == new_schedule
    assert layout.interaction_order != previous_interaction_order

    return

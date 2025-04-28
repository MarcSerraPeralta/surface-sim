import pytest

from surface_sim.layouts.operations import check_overlap_layouts
from surface_sim.layouts import rot_surface_code


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

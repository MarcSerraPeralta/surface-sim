import pytest

from surface_sim.layouts import rot_surface_code, set_coords


def test_set_coords():
    layout = rot_surface_code(distance=3)

    with pytest.raises(Exception):
        set_coords(layout)

    return

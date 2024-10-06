from surface_sim import Layout
from surface_sim.layouts import rot_surf_code, rot_surf_code_rectangle


def test_rot_surf_code():
    layout = rot_surf_code(distance=3)

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 9
    assert len(layout.get_qubits(role="anc")) == 8

    return


def test_rot_surf_code_rectangle():
    layout = rot_surf_code_rectangle(distance_x=3, distance_z=4)

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 12
    assert len(layout.get_qubits(role="anc")) == 11
    assert len(layout.log_z) == 4
    assert len(layout.log_x) == 3

    return

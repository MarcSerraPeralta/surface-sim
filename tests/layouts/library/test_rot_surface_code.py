import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import rot_surf_code, rot_surf_code_rectangle
from surface_sim.layouts import plot


def test_rot_surf_code(show_figures):
    layout = rot_surf_code(distance=5)

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 25
    assert len(layout.get_qubits(role="anc")) == 24
    assert len(layout.get_logical_qubits()) == 1
    log_label = layout.get_logical_qubits()[0]
    assert len(layout.log_z[log_label]) == 5
    assert len(layout.log_x[log_label]) == 5

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_rot_surf_code_rectangle(show_figures):
    layout = rot_surf_code_rectangle(distance_x=3, distance_z=4)

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 12
    assert len(layout.get_qubits(role="anc")) == 11
    assert len(layout.get_logical_qubits()) == 1
    log_label = layout.get_logical_qubits()[0]
    assert len(layout.log_z[log_label]) == 4
    assert len(layout.log_x[log_label]) == 3

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return

import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import unrot_surf_code, unrot_surf_code_rectangle
from surface_sim.layouts import plot


def test_unrot_surf_code(show_figures):
    layout = unrot_surf_code(distance=5)

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 5**2 + 4**2
    assert len(layout.get_qubits(role="anc")) == (5 + 4) * 4 + 4
    assert len(layout.get_logical_qubits()) == 1
    log_label = layout.get_logical_qubits()[0]
    assert len(layout.log_z[log_label]) == 5
    assert len(layout.log_x[log_label]) == 5

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout)
        plt.show()

    return


def test_unrot_surf_code_rectangle(show_figures):
    layout = unrot_surf_code_rectangle(distance_x=3, distance_z=4)

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 4 * 3 + 3 * 2
    assert len(layout.get_qubits(role="anc")) == (4 + 3) * 2 + 3
    assert len(layout.get_logical_qubits()) == 1
    log_label = layout.get_logical_qubits()[0]
    assert len(layout.log_z[log_label]) == 4
    assert len(layout.log_x[log_label]) == 3

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout)
        plt.show()

    return

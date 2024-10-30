import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import rot_surface_code, rot_surface_code_rectangle
from surface_sim.layouts import plot


def test_rot_surface_code(show_figures):
    layout = rot_surface_code(
        distance=5,
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 25
    assert len(layout.get_qubits(role="anc")) == 24
    assert len(layout.get_logical_qubits()) == 1
    log_label = layout.get_logical_qubits()[0]
    assert len(layout.log_z[log_label]) == 5
    assert len(layout.log_x[log_label]) == 5
    assert min(layout.get_inds(layout.get_qubits())) == 10

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_rot_surface_code_rectangle(show_figures):
    layout = rot_surface_code_rectangle(
        distance_x=3,
        distance_z=4,
        init_point=(2, 3),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=11,
    )

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 12
    assert len(layout.get_qubits(role="anc")) == 11
    assert len(layout.get_logical_qubits()) == 1
    log_label = layout.get_logical_qubits()[0]
    assert len(layout.log_z[log_label]) == 4
    assert len(layout.log_x[log_label]) == 3
    assert min(layout.get_inds(layout.get_qubits())) == 11

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return

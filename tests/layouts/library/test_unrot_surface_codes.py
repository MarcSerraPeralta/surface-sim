import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import unrot_surface_code, unrot_surface_code_rectangle
from surface_sim.layouts import plot


def test_unrot_surface_code(show_figures):
    layout = unrot_surface_code(
        distance=5,
        init_point=(-2, -3),
        init_data_qubit_id=3,
        init_zanc_qubit_id=5,
        init_xanc_qubit_id=4,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 5**2 + 4**2
    assert len(layout.get_qubits(role="anc")) == (5 + 4) * 4 + 4
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


def test_unrot_surface_code_rectangle(show_figures):
    layout = unrot_surface_code_rectangle(
        distance_x=3,
        distance_z=4,
        init_point=(2, 3),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=11,
    )

    assert isinstance(layout, Layout)
    assert len(layout.get_qubits(role="data")) == 4 * 3 + 3 * 2
    assert len(layout.get_qubits(role="anc")) == (4 + 3) * 2 + 3
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

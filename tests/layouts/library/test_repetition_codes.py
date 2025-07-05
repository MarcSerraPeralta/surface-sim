import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import (
    repetition_code,
)
from surface_sim.layouts import plot


def test_rot_surface_code_x_type(show_figures):
    layout = repetition_code(
        distance=5,
        stab_type="x_type",
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.data_qubits) == 5
    assert len(layout.anc_qubits) == 4
    assert len(layout.logical_qubits) == 1
    log_label = layout.logical_qubits[0]
    assert len(layout.logical_param("log_z", log_label)) == 5
    assert len(layout.logical_param("log_x", log_label)) == 1
    assert min(layout.get_inds(layout.qubits)) == 10

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_rot_surface_code_z_type(show_figures):
    layout = repetition_code(
        distance=5,
        stab_type="z_type",
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.data_qubits) == 5
    assert len(layout.anc_qubits) == 4
    assert len(layout.logical_qubits) == 1
    log_label = layout.logical_qubits[0]
    assert len(layout.logical_param("log_z", log_label)) == 1
    assert len(layout.logical_param("log_x", log_label)) == 5
    assert min(layout.get_inds(layout.qubits)) == 10

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return

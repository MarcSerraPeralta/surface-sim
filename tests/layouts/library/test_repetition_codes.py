import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import (
    repetition_code,
    repetition_stability,
)
from surface_sim.layouts import plot


def test_repetition_code_x_type(show_figures):
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
    assert layout.anc_qubits[0][0] == "X"
    assert len(layout.logical_qubits) == 1
    log_label = layout.logical_qubits[0]
    assert len(layout.logical_param("log_z", log_label)) == 5
    assert len(layout.logical_param("log_x", log_label)) == 1
    assert min(layout.get_inds(layout.qubits)) == 10
    assert "steps" in layout.interaction_order

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_repetition_code_z_type(show_figures):
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
    assert layout.anc_qubits[0][0] == "Z"
    assert len(layout.logical_qubits) == 1
    log_label = layout.logical_qubits[0]
    assert len(layout.logical_param("log_z", log_label)) == 1
    assert len(layout.logical_param("log_x", log_label)) == 5
    assert min(layout.get_inds(layout.qubits)) == 10
    assert "steps" in layout.interaction_order

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_repetition_stability_z_type(show_figures):
    layout = repetition_stability(
        stab_type="z_type",
        num_stabs=4,
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.data_qubits) == 3
    assert len(layout.anc_qubits) == 4
    assert layout.anc_qubits[0][0] == "Z"
    assert len(layout.logical_qubits) == 0
    assert len(layout.observables) == 1
    assert len(layout.observable_definition("O0")) == 4
    assert min(layout.get_inds(layout.qubits)) == 10
    assert "steps" in layout.interaction_order

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_repetition_stability_x_type(show_figures):
    layout = repetition_stability(
        stab_type="x_type",
        num_stabs=7,
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.data_qubits) == 6
    assert len(layout.anc_qubits) == 7
    assert layout.anc_qubits[0][0] == "X"
    assert len(layout.logical_qubits) == 0
    assert len(layout.observables) == 1
    assert len(layout.observable_definition("O0")) == 7
    assert min(layout.get_inds(layout.qubits)) == 10
    assert "steps" in layout.interaction_order

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return

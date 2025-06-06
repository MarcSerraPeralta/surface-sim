import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import (
    unrot_surface_code,
    unrot_surface_code_rectangle,
    unrot_surface_codes,
)
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
    assert len(layout.data_qubits) == 5**2 + 4**2
    assert len(layout.anc_qubits) == (5 + 4) * 4 + 4
    assert len(layout.logical_qubits) == 1
    log_label = layout.logical_qubits[0]
    assert len(layout.logical_param("log_z", log_label)) == 5
    assert len(layout.logical_param("log_x", log_label)) == 5
    assert min(layout.get_inds(layout.qubits)) == 10

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
    assert len(layout.data_qubits) == 4 * 3 + 3 * 2
    assert len(layout.anc_qubits) == (4 + 3) * 2 + 3
    assert len(layout.logical_qubits) == 1
    log_label = layout.logical_qubits[0]
    assert len(layout.logical_param("log_z", log_label)) == 4
    assert len(layout.logical_param("log_x", log_label)) == 3
    assert min(layout.get_inds(layout.qubits)) == 11

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return


def test_unrot_surface_codes():
    layouts = unrot_surface_codes(2, distance=3)

    assert isinstance(layouts, list)
    assert len(layouts) == 2

    layout_0, layout_1 = layouts
    assert set(
        [
            "log_fold_trans_h_L0",
            "log_fold_trans_s_L0",
            "log_x_L0",
            "log_z_L0",
            "idle_L0",
            "log_trans_cnot_L0_L1",
            "log_trans_cnot_L1_L0",
        ]
    ) < set(layout_0.to_dict()["layout"][0].keys())
    assert set(
        [
            "log_fold_trans_h_L1",
            "log_fold_trans_s_L1",
            "log_x_L1",
            "log_z_L1",
            "idle_L1",
            "log_trans_cnot_L0_L1",
            "log_trans_cnot_L1_L0",
        ]
    ) < set(layout_1.to_dict()["layout"][0].keys())

    return

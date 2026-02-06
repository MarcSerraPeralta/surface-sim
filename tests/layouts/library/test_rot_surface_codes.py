import matplotlib.pyplot as plt

from surface_sim import Layout
from surface_sim.layouts import (
    plot,
    rot_surface_code,
    rot_surface_code_rectangle,
    rot_surface_code_rectangles,
    rot_surface_codes,
    rot_surface_stability_rectangle,
)


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
    assert len(layout.data_qubits) == 25
    assert len(layout.anc_qubits) == 24
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
    assert len(layout.data_qubits) == 12
    assert len(layout.anc_qubits) == 11
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


def test_rot_surface_codes():
    layouts = rot_surface_codes(2, distance=3)

    assert isinstance(layouts, list)
    assert len(layouts) == 2

    layout_0, layout_1 = layouts
    assert set(
        [
            "log_x_L0",
            "log_z_L0",
            "idle_L0",
            "log_trans_cnot_L0_L1",
            "log_trans_cnot_L1_L0",
        ]
    ) < set(layout_0.to_dict()["layout"][0].keys())
    assert set(
        [
            "log_x_L1",
            "log_z_L1",
            "idle_L1",
            "log_trans_cnot_L0_L1",
            "log_trans_cnot_L1_L0",
        ]
    ) < set(layout_1.to_dict()["layout"][0].keys())

    return


def test_rot_surface_code_rectangles():
    layouts = rot_surface_code_rectangles(2, distance=3)

    assert isinstance(layouts, list)
    assert len(layouts) == 2

    layout_0, layout_1 = layouts
    assert set(
        [
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
            "log_fold_trans_s_L1",
            "log_x_L1",
            "log_z_L1",
            "idle_L1",
            "log_trans_cnot_L0_L1",
            "log_trans_cnot_L1_L0",
        ]
    ) < set(layout_1.to_dict()["layout"][0].keys())

    return


def test_rot_surface_stability_rectangle_z_type(show_figures):
    layout = rot_surface_stability_rectangle(
        width=2,
        height=3,
        stab_type="z_type",
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )
    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    assert isinstance(layout, Layout)
    assert len(layout.data_qubits) == 6
    assert len(layout.anc_qubits) == 7
    assert len(layout.logical_qubits) == 0
    assert len(layout.observables) == 1
    assert len(layout.observable_definition("O0")) == 6
    assert min(layout.get_inds(layout.qubits)) == 10

    return


def test_rot_surface_stability_rectangle_x_type(show_figures):
    layout = rot_surface_stability_rectangle(
        width=4,
        height=4,
        stab_type="x_type",
        init_point=(3, 4),
        init_data_qubit_id=2,
        init_zanc_qubit_id=4,
        init_xanc_qubit_id=5,
        init_ind=10,
    )

    assert isinstance(layout, Layout)
    assert len(layout.data_qubits) == 16
    assert len(layout.anc_qubits) == 17
    assert len(layout.logical_qubits) == 0
    assert len(layout.observables) == 1
    assert len(layout.observable_definition("O0")) == 12
    assert min(layout.get_inds(layout.qubits)) == 10

    if show_figures:
        _, ax = plt.subplots()
        plot(ax, layout, stim_orientation=False)
        plt.show()

    return

import stim

from surface_sim.layouts import rot_surf_code, rot_surf_code_rectangle

from surface_sim.experiments.rot_surface_code_css import (
    memory_experiment,
    repeated_s_experiment,
)
from surface_sim.models import NoiselessModel
from surface_sim import Detectors
from surface_sim.log_gates.rot_surface_code_css import set_trans_s


def test_memory_experiment():
    layout = rot_surf_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = memory_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_rounds=10,
        anc_reset=False,
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    assert isinstance(circuit, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    _ = circuit.detector_error_model(allow_gauge_detectors=True)

    return


def test_repeated_s_experiment():
    layout = rot_surf_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = repeated_s_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_s_gates=4,
        num_rounds_per_gate=2,
        anc_reset=False,
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    assert isinstance(circuit, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    _ = circuit.detector_error_model(allow_gauge_detectors=True)

    return


def test_memory_experiment_anc_detectors():
    layout = rot_surf_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = memory_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_rounds=10,
        anc_reset=False,
        anc_detectors=["X1"],
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    num_anc = len(layout.get_qubits(role="anc"))
    num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
    assert circuit.num_detectors == 10 * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == 10 + 1

    return


def test_repeated_s_experiment_anc_detectors():
    layout = rot_surf_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    detectors = Detectors(layout.get_qubits(role="anc"), frame="1")
    circuit = repeated_s_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_s_gates=4,
        num_rounds_per_gate=2,
        anc_reset=False,
        anc_detectors=["X1"],
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    num_anc = len(layout.get_qubits(role="anc"))
    num_anc_x = len(layout.get_qubits(role="anc", stab_type="x_type"))
    assert circuit.num_detectors == (1 + 4 * 2) * num_anc + num_anc_x

    non_zero_dets = []
    for instr in circuit.flattened():
        if instr.name == "DETECTOR" and len(instr.targets_copy()) != 0:
            non_zero_dets.append(instr)

    assert len(non_zero_dets) == 1 + 4 * 2 + 1

    return

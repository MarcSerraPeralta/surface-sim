import stim

from surface_sim.layouts import rot_surf_code, rot_surf_code_rectangle

from surface_sim.experiments.surface_code_css import (
    memory_experiment,
    repeated_s_experiment,
)
from surface_sim.models import NoiselessModel
from surface_sim import Detectors
from surface_sim.log_gates.surface_code_css import set_trans_s


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
        meas_reset=False,
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    assert isinstance(circuit, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    circuit.diagram(type="detslice-with-ops")

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
        meas_reset=False,
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    assert isinstance(circuit, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    circuit.diagram(type="detslice-with-ops")

    return

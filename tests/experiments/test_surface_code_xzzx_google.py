import stim

from surface_sim.layouts import rot_surf_code

from surface_sim.experiments.surface_code_xzzx_google import memory_experiment
from surface_sim.models import NoiselessModel
from surface_sim import Detectors


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
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    assert isinstance(circuit, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    circuit.diagram(type="detslice-with-ops")

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
        anc_detectors=["X1"],
        data_init={q: 0 for q in layout.get_qubits(role="data")},
        rot_basis=True,
    )

    assert circuit.num_detectors == 10 + 1

    return

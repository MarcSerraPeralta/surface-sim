import stim

from qec_util.layouts import rot_surf_code

from surface_sim.experiments.surface_code_xzzx_google import memory_experiment
from surface_sim.models import NoiselessModel


def test_memory_experiment():
    layout = rot_surf_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    circuit = memory_experiment(
        model=model,
        layout=layout,
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

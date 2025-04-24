import numpy as np

from surface_sim.layouts import rot_surface_code
from surface_sim.util import sample_memory_experiment
from surface_sim.models import NoiselessModel
from surface_sim.experiments.rot_surface_code_xzzx import memory_experiment
from surface_sim import Detectors


def test_sample_memory_experiment():
    layout = rot_surface_code(distance=3)
    model = NoiselessModel(layout.qubit_inds())
    detectors = Detectors(layout.anc_qubits, frame="pre-gate")
    circuit = memory_experiment(
        model=model,
        layout=layout,
        detectors=detectors,
        num_rounds=10,
        anc_reset=False,
        data_init={q: 0 for q in layout.data_qubits},
        rot_basis=True,
    )

    dataset = sample_memory_experiment(
        layout=layout,
        experiment=circuit,
        num_shots=100,
        num_rounds=10,
        seed=123,
    )

    assert set(dataset.data_vars) == {
        "ideal_anc_meas",
        "ideal_data_meas",
        "data_meas",
        "anc_meas",
    }
    assert set(dataset.coords) == {
        "qec_round",
        "seed",
        "shot",
        "anc_qubit",
        "data_qubit",
    }
    assert (dataset.qec_round.values == np.arange(1, 10 + 1)).all()
    assert (dataset.shot.values == np.arange(100)).all()
    assert (dataset.anc_qubit.values == np.array(layout.anc_qubits)).all()
    assert (dataset.data_qubit.values == np.array(layout.data_qubits)).all()
    assert dataset.seed == 123

    return

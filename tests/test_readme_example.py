from surface_sim.layouts import rot_surface_code
from surface_sim.models import CircuitNoiseModel
from surface_sim.setup import CircuitNoiseSetup
from surface_sim import Detectors
from surface_sim.experiments.rot_surface_code_css import memory_experiment


def test_README_example():
    # prepare the layout, model, and detectors objects
    layout = rot_surface_code(distance=3)

    qubit_inds = layout.qubit_inds()
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    setup = CircuitNoiseSetup()

    model = CircuitNoiseModel(setup, qubit_inds)

    detectors = Detectors(anc_qubits, frame="1")

    # create a memory experiment
    NUM_ROUNDS = 10
    DATA_INIT = {q: 0 for q in data_qubits}
    ROT_BASIS = True  # X basis
    MEAS_RESET = True  # reset after ancilla measurements
    PROB = 1e-5

    setup.set_var_param("prob", PROB)
    stim_circuit = memory_experiment(
        model, layout, detectors, NUM_ROUNDS, DATA_INIT, ROT_BASIS, MEAS_RESET
    )

    stim_circuit.detector_error_model(allow_gauge_detectors=True)

    return

def test_readme_example_memory_experiment():
    from surface_sim.layouts import rot_surface_code
    from surface_sim.models import CircuitNoiseModel
    from surface_sim.setup import CircuitNoiseSetup
    from surface_sim import Detectors
    from surface_sim.experiments.rot_surface_code_css import memory_experiment

    # prepare the layout, model, and detectors objects
    layout = rot_surface_code(distance=3)

    qubit_inds = layout.qubit_inds()
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    setup = CircuitNoiseSetup()
    model = CircuitNoiseModel(setup, qubit_inds)
    detectors = Detectors(anc_qubits, frame="pre-gate")

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


def test_readme_example_arbitrary_circuit():
    import stim

    from surface_sim.setup import CircuitNoiseSetup
    from surface_sim.models import CircuitNoiseModel
    from surface_sim import Detectors
    from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
    from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
    from surface_sim.layouts import unrot_surface_codes

    circuit = stim.Circuit(
        """
        R 0 1
        TICK
        CNOT 0 1
        TICK
        S 0
        I 1
        TICK
        S 0
        H 1
        TICK
        M 0
        MX 1
        """
    )

    layouts = unrot_surface_codes(circuit.num_qubits, distance=3)

    # merge qubit indicies, coordinates, ... of all layouts
    qubit_inds, anc_coords, anc_qubits = {}, {}, []
    for layout in layouts:
        qubit_inds.update(layout.qubit_inds())
        anc_qubits += layout.get_qubits(role="anc")
        anc_coords.update(layout.anc_coords())

    setup = CircuitNoiseSetup()
    setup.set_var_param("prob", 1e-3)
    model = CircuitNoiseModel(setup=setup, qubit_inds=qubit_inds)
    detectors = Detectors(anc_qubits, frame="pre-gate", anc_coords=anc_coords)

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    stim_circuit = experiment_from_schedule(schedule, model, detectors, anc_reset=True)

    stim_circuit.detector_error_model(allow_gauge_detectors=True)

    return

import pytest
import stim

from surface_sim import Detectors
from surface_sim.circuit_blocks.decorators import noiseless
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.experiments import experiment_from_schedule, schedule_from_circuit
from surface_sim.experiments.arbitrary_experiment import schedule_from_instructions
from surface_sim.layouts import unrot_surface_codes
from surface_sim.models import CircuitNoiseModel, NoiselessModel


def test_schedule_from_circuit():
    layouts = unrot_surface_codes(4, distance=3)
    circuit = stim.Circuit(
        """
        R 0 1 2
        TICK
        X 0
        TICK
        CX 0 1
        """
    )

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)

    assert len(schedule) == 5

    expected_schedule = [
        [
            (gate_to_iterator["R"], layouts[0]),
            (gate_to_iterator["R"], layouts[1]),
            (gate_to_iterator["R"], layouts[2]),
        ],
        [
            (gate_to_iterator["TICK"], layouts[0]),
            (gate_to_iterator["TICK"], layouts[1]),
            (gate_to_iterator["TICK"], layouts[2]),
        ],
        [
            (gate_to_iterator["X"], layouts[0]),
            (gate_to_iterator["I"], layouts[1]),
            (gate_to_iterator["I"], layouts[2]),
        ],
        [
            (gate_to_iterator["TICK"], layouts[0]),
            (gate_to_iterator["TICK"], layouts[1]),
            (gate_to_iterator["TICK"], layouts[2]),
        ],
        [
            (gate_to_iterator["CX"], layouts[0], layouts[1]),
            (gate_to_iterator["I"], layouts[2]),
        ],
    ]

    assert expected_schedule == schedule

    return


def test_schedule_from_instructions():
    layouts = unrot_surface_codes(4, distance=3)
    instructions = [(gate_to_iterator["X"], layouts[0])]

    with pytest.raises(ValueError):
        _ = schedule_from_instructions(instructions)

    instructions = [
        (gate_to_iterator["R"], layouts[0]),
        (gate_to_iterator["M"], layouts[0]),
        (gate_to_iterator["X"], layouts[0]),
    ]
    with pytest.raises(ValueError):
        _ = schedule_from_instructions(instructions)

    instructions = [
        (gate_to_iterator["R"], layouts[0]),
        (gate_to_iterator["R"], layouts[1]),
        (gate_to_iterator["R"], layouts[2]),
        (gate_to_iterator["R"], layouts[3]),
        (gate_to_iterator["TICK"],),
        (gate_to_iterator["X"], layouts[0]),
        (gate_to_iterator["M"], layouts[1]),
        (gate_to_iterator["TICK"],),
        (gate_to_iterator["CX"], layouts[2], layouts[3]),
    ]
    schedule = schedule_from_instructions(instructions)

    expected_schedule = [
        [
            (gate_to_iterator["R"], layouts[0]),
            (gate_to_iterator["R"], layouts[1]),
            (gate_to_iterator["R"], layouts[2]),
            (gate_to_iterator["R"], layouts[3]),
        ],
        [
            (gate_to_iterator["TICK"], layouts[0]),
            (gate_to_iterator["TICK"], layouts[1]),
            (gate_to_iterator["TICK"], layouts[2]),
            (gate_to_iterator["TICK"], layouts[3]),
        ],
        [
            (gate_to_iterator["X"], layouts[0]),
            (gate_to_iterator["M"], layouts[1]),
            (gate_to_iterator["I"], layouts[2]),
            (gate_to_iterator["I"], layouts[3]),
        ],
        [
            (gate_to_iterator["TICK"], layouts[0]),
            (gate_to_iterator["TICK"], layouts[2]),
            (gate_to_iterator["TICK"], layouts[3]),
        ],
        [
            (gate_to_iterator["CX"], layouts[2], layouts[3]),
            (gate_to_iterator["I"], layouts[0]),
        ],
    ]

    assert schedule == expected_schedule

    instructions = [
        (gate_to_iterator["R"], layouts[0]),
        (gate_to_iterator["TICK"],),
        (gate_to_iterator["TICK"],),
        (gate_to_iterator["M"], layouts[0]),
    ]

    schedule = schedule_from_instructions(instructions)

    expected_schedule = [
        [(gate_to_iterator["R"], layouts[0])],
        [(gate_to_iterator["TICK"], layouts[0])],
        [(gate_to_iterator["TICK"], layouts[0])],
        [(gate_to_iterator["M"], layouts[0])],
    ]

    assert schedule == expected_schedule

    return


def test_experiment_from_schedule():
    layouts = unrot_surface_codes(3, distance=3)

    circuit = stim.Circuit(
        """
        R 0 1
        TICK
        X 1
        I 0
        TICK
        M 0
        I 1
        TICK
        """
    )
    model = NoiselessModel.from_layouts(*layouts)
    detectors = Detectors.from_layouts("pre-gate", *layouts)

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=None
    )

    assert isinstance(experiment, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    dem = circuit.detector_error_model(allow_gauge_detectors=False)

    num_coords = 0
    anc_coords = {}
    for layout in layouts:
        anc_coords |= layout.anc_coords
    anc_coords = {k: list(map(float, v)) for k, v in anc_coords.items()}
    for dem_instr in dem:
        if dem_instr.type == "detector":
            assert dem_instr.args_copy()[:-1] in anc_coords.values()
            num_coords += 1

    assert num_coords == dem.num_detectors

    return


def test_experiment_from_schedule_no_gauge_detectors():
    layouts = unrot_surface_codes(3, distance=3)

    circuit = stim.Circuit(
        """
        R 0 1
        TICK
        X 1
        I 0
        TICK
        M 0
        I 1
        TICK
        """
    )
    model = NoiselessModel.from_layouts(*layouts)
    detectors = Detectors.from_layouts("pre-gate", *layouts, include_gauge_dets=False)

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    experiment = experiment_from_schedule(
        schedule,
        model,
        detectors,
        anc_reset=True,
        anc_detectors=None,
    )

    assert isinstance(experiment, stim.Circuit)

    # check that the detectors and logicals fulfill their
    # conditions by building the stim diagram
    _ = circuit.detector_error_model(allow_gauge_detectors=False)

    return


def test_module_2_operations_in_detectors():
    layouts = unrot_surface_codes(2, distance=3)
    model = CircuitNoiseModel.from_layouts(*layouts)
    model.setup.set_var_param("prob", 1e-3)
    detectors = Detectors.from_layouts("pre-gate", *layouts)

    circuit = stim.Circuit(
        """
        RX 0 1
        TICK
        CX 1 0
        TICK
        S 0 1
        TICK
        CX 0 1
        TICK
        CX 0 1
        TICK
        S 0 1
        TICK
        CX 1 0
        TICK
        MX 0 1
    """
    )
    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
    experiment = experiment_from_schedule(
        schedule,
        model,
        detectors,
        anc_reset=True,
        anc_detectors=None,
    )
    experiment = experiment[:-14]  # remove detectors built from data

    for instr in experiment.flattened():
        if instr.name == "DETECTOR":
            assert len(instr.targets_copy()) <= 3

    return


def test_noiseless_decorator():
    layouts = unrot_surface_codes(2, distance=3)
    model = CircuitNoiseModel.from_layouts(*layouts)
    model.setup.set_var_param("prob", 1e-3)
    detectors = Detectors.from_layouts("pre-gate", *layouts)

    noisy_schedule = [
        [
            (gate_to_iterator["R"], layouts[0]),
            (gate_to_iterator["R"], layouts[1]),
        ],
        [
            (gate_to_iterator["TICK"], layouts[0]),
            (gate_to_iterator["TICK"], layouts[1]),
        ],
        [
            (gate_to_iterator["X"], layouts[0]),
            (gate_to_iterator["MX"], layouts[1]),
        ],
        [
            (gate_to_iterator["TICK"], layouts[0]),
        ],
    ]
    noisy_experiment = experiment_from_schedule(
        noisy_schedule,
        model,
        detectors,
        anc_reset=True,
        anc_detectors=None,
    )

    layouts = unrot_surface_codes(2, distance=3)
    model = CircuitNoiseModel.from_layouts(*layouts)
    model.setup.set_var_param("prob", 1e-3)
    detectors = Detectors.from_layouts("pre-gate", *layouts)

    noiseless_schedule = [
        [
            (noiseless(gate_to_iterator["R"]), layouts[0]),
            (noiseless(gate_to_iterator["R"]), layouts[1]),
        ],
        [
            (noiseless(gate_to_iterator["TICK"]), layouts[0]),
            (noiseless(gate_to_iterator["TICK"]), layouts[1]),
        ],
        [
            (noiseless(gate_to_iterator["X"]), layouts[0]),
            (noiseless(gate_to_iterator["MX"]), layouts[1]),
        ],
        [
            (noiseless(gate_to_iterator["TICK"]), layouts[0]),
        ],
    ]
    noiseless_experiment = experiment_from_schedule(
        noiseless_schedule,
        model,
        detectors,
        anc_reset=True,
        anc_detectors=None,
    )

    assert (
        noiseless_experiment.flow_generators()
        == noisy_experiment.without_noise().flow_generators()
    )
    assert len(noiseless_experiment) == len(noisy_experiment.without_noise())

    return

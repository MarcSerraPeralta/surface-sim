from stim import Circuit, target_rec

from surface_sim import Setup
from surface_sim.models import (
    NoiselessModel,
    DecoherenceNoiseModel,
    CircuitNoiseModel,
    IncomingNoiseModel,
    IncomingDepolNoiseModel,
    PhenomenologicalNoiseModel,
    PhenomenologicalDepolNoiseModel,
    MeasurementNoiseModel,
    SI1000NoiseModel,
)

SETUP = {
    "gate_durations": {
        "X": 1,
        "Z": 1,
        "H": 1,
        "CZ": 1,
        "CNOT": 1,
        "SWAP": 1,
        "M": 10,
        "MX": 10,
        "MY": 10,
        "R": 10,
        "RX": 10,
        "RY": 10,
    },
    "setup": [
        {
            "sq_error_prob": 0.1,
            "tq_error_prob": 0.1,
            "meas_error_prob": 0.1,
            "assign_error_flag": True,
            "assign_error_prob": 0.1,
            "reset_error_prob": 0.1,
            "idle_error_prob": 0.1,
            "extra_idle_meas_or_reset_error_prob": 0.1,
            "T1": 1,
            "T2": 1,
            "symmetric_noise": False,
        },
    ],
}

NOISE_GATES = [
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "X_ERROR",
    "Z_ERROR",
    "Y_ERROR",
]


def test_NoiselessModel():
    model = NoiselessModel(qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert ops == ["X"]

    ops = [o.name for o in model.z_gate(["D1"])]
    assert ops == ["Z"]

    ops = [o.name for o in model.hadamard(["D1"])]
    assert ops == ["H"]

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert ops == ["CZ"]

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert ops == ["SWAP"]

    ops = [o.name for o in model.measure(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.measure_x(["D1"])]
    assert ops == ["MX"]

    ops = [o.name for o in model.measure_y(["D1"])]
    assert ops == ["MY"]

    ops = [o.name for o in model.measure_z(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_z(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_x(["D1"])]
    assert ops == ["RX"]

    ops = [o.name for o in model.reset_y(["D1"])]
    assert ops == ["RY"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    return


def test_PhenomenologicalNoiseModel():
    setup = Setup(SETUP)
    model = PhenomenologicalNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert ops == ["X"]

    ops = [o.name for o in model.z_gate(["D1"])]
    assert ops == ["Z"]

    ops = [o.name for o in model.hadamard(["D1"])]
    assert ops == ["H"]

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert ops == ["CZ"]

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert ops == ["SWAP"]

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_z(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_x(["D1"])]
    assert "MX" in ops
    assert set(NOISE_GATES + ["MX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_y(["D1"])]
    assert "MY" in ops
    assert set(NOISE_GATES + ["MY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_z(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_x(["D1"])]
    assert ops == ["RX"]

    ops = [o.name for o in model.reset_y(["D1"])]
    assert ops == ["RY"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert set(NOISE_GATES) >= set(ops)
    assert len(ops) > 1

    return


def test_PhenomenologicalDepolNoiseModel():
    setup = Setup(SETUP)
    model = PhenomenologicalDepolNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert ops == ["X"]

    ops = [o.name for o in model.z_gate(["D1"])]
    assert ops == ["Z"]

    ops = [o.name for o in model.hadamard(["D1"])]
    assert ops == ["H"]

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert ops == ["CZ"]

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert ops == ["SWAP"]

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_z(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_x(["D1"])]
    assert "MX" in ops
    assert set(NOISE_GATES + ["MX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_y(["D1"])]
    assert "MY" in ops
    assert set(NOISE_GATES + ["MY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_z(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_x(["D1"])]
    assert ops == ["RX"]

    ops = [o.name for o in model.reset_y(["D1"])]
    assert ops == ["RY"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert set(NOISE_GATES) >= set(ops)
    assert len(ops) == 1

    return


def test_MeasurementNoiseModel():
    setup = Setup(SETUP)
    model = MeasurementNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert ops == ["X"]

    ops = [o.name for o in model.z_gate(["D1"])]
    assert ops == ["Z"]

    ops = [o.name for o in model.hadamard(["D1"])]
    assert ops == ["H"]

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert ops == ["CZ"]

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert ops == ["SWAP"]

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_z(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_x(["D1"])]
    assert "MX" in ops
    assert set(NOISE_GATES + ["MX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_y(["D1"])]
    assert "MY" in ops
    assert set(NOISE_GATES + ["MY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_z(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_x(["D1"])]
    assert ops == ["RX"]

    ops = [o.name for o in model.reset_y(["D1"])]
    assert ops == ["RY"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_IncomingNoiseModel():
    setup = Setup(SETUP)
    model = IncomingNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert ops == ["X"]

    ops = [o.name for o in model.z_gate(["D1"])]
    assert ops == ["Z"]

    ops = [o.name for o in model.hadamard(["D1"])]
    assert ops == ["H"]

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert ops == ["CZ"]

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert ops == ["SWAP"]

    ops = [o.name for o in model.measure(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.measure_z(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.measure_x(["D1"])]
    assert ops == ["MX"]

    ops = [o.name for o in model.measure_y(["D1"])]
    assert ops == ["MY"]

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_z(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_x(["D1"])]
    assert ops == ["RX"]

    ops = [o.name for o in model.reset_y(["D1"])]
    assert ops == ["RY"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert set(NOISE_GATES) >= set(ops)
    assert len(ops) > 1

    return


def test_IncomingDepolNoiseModel():
    setup = Setup(SETUP)
    model = IncomingDepolNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert ops == ["X"]

    ops = [o.name for o in model.z_gate(["D1"])]
    assert ops == ["Z"]

    ops = [o.name for o in model.hadamard(["D1"])]
    assert ops == ["H"]

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert ops == ["CZ"]

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert ops == ["SWAP"]

    ops = [o.name for o in model.measure(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.measure_z(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.measure_x(["D1"])]
    assert ops == ["MX"]

    ops = [o.name for o in model.measure_y(["D1"])]
    assert ops == ["MY"]

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_z(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.reset_x(["D1"])]
    assert ops == ["RX"]

    ops = [o.name for o in model.reset_y(["D1"])]
    assert ops == ["RY"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert set(NOISE_GATES) >= set(ops)
    assert len(ops) == 1

    return


def test_DecoherentNoiseModel():
    setup = Setup(SETUP)
    model = DecoherenceNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert "X" in ops
    assert set(NOISE_GATES + ["X"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.z_gate(["D1"])]
    assert "Z" in ops
    assert set(NOISE_GATES + ["Z"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.hadamard(["D1"])]
    assert "H" in ops
    assert set(NOISE_GATES + ["H"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert "CZ" in ops
    assert set(NOISE_GATES + ["CZ"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert "CX" in ops
    assert set(NOISE_GATES + ["CX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert "SWAP" in ops
    assert set(NOISE_GATES + ["SWAP"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_z(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_x(["D1"])]
    assert "MX" in ops
    assert set(NOISE_GATES + ["MX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_y(["D1"])]
    assert "MY" in ops
    assert set(NOISE_GATES + ["MY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_z(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_x(["D1"])]
    assert "RX" in ops
    assert set(NOISE_GATES + ["RX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_y(["D1"])]
    assert "RY" in ops
    assert set(NOISE_GATES + ["RY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    # check that extra idling is added if the gate durations do not match
    model.new_circuit()
    circ = Circuit()
    circ += model.reset(["D1"])
    circ += model.hadamard(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 3

    model.new_circuit()
    circ = Circuit()
    circ += model.idle(["D1"])
    circ += model.idle(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 0

    model.new_circuit()
    circ = Circuit()
    circ += model.hadamard(["D1"])
    circ += model.hadamard(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 2

    return


def test_SI1000NoiseModel():
    setup = Setup(SETUP)
    model = SI1000NoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert "X" in ops
    assert set(NOISE_GATES + ["X"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.z_gate(["D1"])]
    assert "Z" in ops
    assert set(NOISE_GATES + ["Z"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.hadamard(["D1"])]
    assert "H" in ops
    assert set(NOISE_GATES + ["H"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert "CZ" in ops
    assert set(NOISE_GATES + ["CZ"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert "CX" in ops
    assert set(NOISE_GATES + ["CX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert "SWAP" in ops
    assert set(NOISE_GATES + ["SWAP"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_z(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_x(["D1"])]
    assert "MX" in ops
    assert set(NOISE_GATES + ["MX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_y(["D1"])]
    assert "MY" in ops
    assert set(NOISE_GATES + ["MY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_z(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_x(["D1"])]
    assert "RX" in ops
    assert set(NOISE_GATES + ["RX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_y(["D1"])]
    assert "RY" in ops
    assert set(NOISE_GATES + ["RY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.idle(["D1"])]
    assert set(NOISE_GATES + ["I"]) >= set(ops)
    assert len(ops) > 0

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    # check extra noise channels when doing measurement or resets
    model.new_circuit()
    circ = Circuit()
    circ += model.idle(["D1"])
    circ += model.measure(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 3

    model.new_circuit()
    circ = Circuit()
    circ += model.idle(["D1"])
    circ += model.reset(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 3

    model.new_circuit()
    circ = Circuit()
    circ += model.idle(["D1"])
    circ += model.idle(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 2

    return


def test_CircuitNoiseModel():
    setup = Setup(SETUP)
    model = CircuitNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert "X" in ops
    assert set(NOISE_GATES + ["X"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.z_gate(["D1"])]
    assert "Z" in ops
    assert set(NOISE_GATES + ["Z"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.hadamard(["D1"])]
    assert "H" in ops
    assert set(NOISE_GATES + ["H"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert "CZ" in ops
    assert set(NOISE_GATES + ["CZ"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert "CX" in ops
    assert set(NOISE_GATES + ["CX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert "SWAP" in ops
    assert set(NOISE_GATES + ["SWAP"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_z(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_x(["D1"])]
    assert "MX" in ops
    assert set(NOISE_GATES + ["MX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.measure_y(["D1"])]
    assert "MY" in ops
    assert set(NOISE_GATES + ["MY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_z(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_x(["D1"])]
    assert "RX" in ops
    assert set(NOISE_GATES + ["RX"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset_y(["D1"])]
    assert "RY" in ops
    assert set(NOISE_GATES + ["RY"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.idle(["D1"])]
    assert set(NOISE_GATES + ["I"]) >= set(ops)
    assert len(ops) > 0

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_model_meas_order():
    setup = Setup(SETUP)
    qubit_inds = {"D1": 1, "D2": 2}
    models = [
        CircuitNoiseModel(setup, qubit_inds=qubit_inds),
        NoiselessModel(qubit_inds=qubit_inds),
        DecoherenceNoiseModel(setup, qubit_inds=qubit_inds),
        IncomingNoiseModel(setup, qubit_inds=qubit_inds),
        PhenomenologicalNoiseModel(setup, qubit_inds=qubit_inds),
        SI1000NoiseModel(setup, qubit_inds=qubit_inds),
    ]

    for model in models:
        circuit = Circuit()
        for instr in model.measure(["D1", "D2"]):
            circuit.append(instr)
        for instr in model.measure(["D1"]):
            circuit.append(instr)

        assert model.meas_target("D1", -2) == target_rec(-3)

        model.new_circuit()
        circuit = Circuit()
        for instr in model.measure(["D1"]):
            circuit.append(instr)
        for instr in model.measure(["D1"]):
            circuit.append(instr)

        assert model.meas_target("D1", -2) == target_rec(-2)

    return

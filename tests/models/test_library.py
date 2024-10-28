from stim import Circuit, target_rec

from surface_sim import Setup
from surface_sim.models import (
    NoiselessModel,
    DecoherenceNoiseModel,
    ExperimentalNoiseModel,
    CircuitNoiseModel,
    IncomingNoiseModel,
    PhenomenologicalNoiseModel,
)

SETUP = {
    "gate_durations": {
        "X": 1,
        "Z": 1,
        "H": 1,
        "CZ": 1,
        "CNOT": 1,
        "M": 1,
        "R": 1,
    },
    "setup": [
        {
            "sq_error_prob": 0.1,
            "cz_error_prob": 0.1,
            "meas_error_prob": 0.1,
            "assign_error_flag": True,
            "assign_error_prob": 0.1,
            "reset_error_prob": 0.1,
            "idle_error_prob": 0.1,
            "T1": 1,
            "T2": 1,
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

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.measure(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

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

    ops = [o.name for o in model.cnot(["D1", "D2"])]
    assert ops == ["CX"]

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert set(NOISE_GATES) >= set(ops)
    assert len(ops) > 1

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

    ops = [o.name for o in model.measure(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert set(NOISE_GATES) >= set(ops)
    assert len(ops) > 1

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

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.idle(["D1"], duration=1)]
    assert set(NOISE_GATES + ["I"]) >= set(ops)
    assert len(ops) > 0

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_ExperimentalNoiseModel():
    setup = Setup(SETUP)
    model = ExperimentalNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

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

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)
    assert len(ops) > 1

    ops = [o.name for o in model.idle(["D1"], duration=1)]
    assert set(NOISE_GATES + ["I"]) >= set(ops)
    assert len(ops) > 0

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

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


def test_model_meas_roder():
    setup = Setup(SETUP)
    qubit_inds = {"D1": 1, "D2": 2}
    models = [
        CircuitNoiseModel(setup, qubit_inds=qubit_inds),
        NoiselessModel(qubit_inds=qubit_inds),
        DecoherenceNoiseModel(setup, qubit_inds=qubit_inds),
        IncomingNoiseModel(setup, qubit_inds=qubit_inds),
        PhenomenologicalNoiseModel(setup, qubit_inds=qubit_inds),
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

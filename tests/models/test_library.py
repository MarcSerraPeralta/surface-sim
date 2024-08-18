from surface_sim import Setup
from surface_sim.models import (
    NoiselessModel,
    DecoherenceNoiseModel,
    CircuitNoiseModel,
)

SETUP = {
    "gate_durations": {
        "X": 1,
        "Z": 1,
        "H": 1,
        "CZ": 1,
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

    ops = [o.name for o in model.measure(["D1"])]
    assert ops == ["M"]

    ops = [o.name for o in model.reset(["D1"])]
    assert ops == ["R"]

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    return


def test_DecoherentNoiseModel():
    setup = Setup(SETUP)
    model = DecoherenceNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert "X" in ops
    assert set(NOISE_GATES + ["X"]) >= set(ops)

    ops = [o.name for o in model.z_gate(["D1"])]
    assert "Z" in ops
    assert set(NOISE_GATES + ["Z"]) >= set(ops)

    ops = [o.name for o in model.hadamard(["D1"])]
    assert "H" in ops
    assert set(NOISE_GATES + ["H"]) >= set(ops)

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert "CZ" in ops
    assert set(NOISE_GATES + ["CZ"]) >= set(ops)

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)

    ops = [o.name for o in model.idle(["D1"], duration=1)]
    assert set(NOISE_GATES + ["I"]) >= set(ops)

    return


def test_CircuitNoiseModel():
    setup = Setup(SETUP)
    model = CircuitNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    ops = [o.name for o in model.x_gate(["D1"])]
    assert "X" in ops
    assert set(NOISE_GATES + ["X"]) >= set(ops)

    ops = [o.name for o in model.z_gate(["D1"])]
    assert "Z" in ops
    assert set(NOISE_GATES + ["Z"]) >= set(ops)

    ops = [o.name for o in model.hadamard(["D1"])]
    assert "H" in ops
    assert set(NOISE_GATES + ["H"]) >= set(ops)

    ops = [o.name for o in model.cphase(["D1", "D2"])]
    assert "CZ" in ops
    assert set(NOISE_GATES + ["CZ"]) >= set(ops)

    ops = [o.name for o in model.measure(["D1"])]
    assert "M" in ops
    assert set(NOISE_GATES + ["M"]) >= set(ops)

    ops = [o.name for o in model.reset(["D1"])]
    assert "R" in ops
    assert set(NOISE_GATES + ["R"]) >= set(ops)

    ops = [o.name for o in model.idle(["D1"])]
    assert set(NOISE_GATES + ["I"]) >= set(ops)

    return

from stim import Circuit, target_rec

from surface_sim import Setup
from surface_sim.models import (
    NoiselessModel,
    T1T2NoiseModel,
    CircuitNoiseModel,
    IncomingNoiseModel,
    IncomingDepolNoiseModel,
    PhenomenologicalNoiseModel,
    PhenomenologicalDepolNoiseModel,
    MeasurementNoiseModel,
    SI1000NoiseModel,
    BiasedCircuitNoiseModel,
    MovableQubitsCircuitNoiseModel,
)
from surface_sim.setups import SQ_GATES, SQ_RESETS, SQ_MEASUREMENTS, TQ_GATES

SETUP = {
    "gate_durations": {n: 1 for n in (SQ_GATES | TQ_GATES).values()}
    | {n: 10 for n in (SQ_RESETS | SQ_MEASUREMENTS).values()},
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
            "biased_pauli": "X",
            "biased_factor": 0,
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

    SQ_OPS = SQ_GATES | SQ_RESETS | SQ_MEASUREMENTS
    for name in SQ_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert ops == [SQ_OPS[name]]

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert ops == [TQ_GATES[name]]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_PhenomenologicalNoiseModel():
    setup = Setup(SETUP)
    model = PhenomenologicalNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_NOISELESS_OPS = SQ_GATES | SQ_RESETS
    for name in SQ_NOISELESS_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert ops == [SQ_NOISELESS_OPS[name]]

    for name in SQ_MEASUREMENTS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_MEASUREMENTS[name] in ops
        assert set(NOISE_GATES + [SQ_MEASUREMENTS[name]]) >= set(ops)
        # noise before the measurement
        assert ops.index(SQ_MEASUREMENTS[name]) == len(ops) - 1
        assert len(ops) > 1

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert ops == [TQ_GATES[name]]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert ops == ["X_ERROR", "Z_ERROR"]

    return


def test_PhenomenologicalDepolNoiseModel():
    setup = Setup(SETUP)
    model = PhenomenologicalDepolNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_NOISELESS_OPS = SQ_GATES | SQ_RESETS
    for name in SQ_NOISELESS_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert ops == [SQ_NOISELESS_OPS[name]]

    ops = [o.name for o in model.idle(["D1"])]
    assert set(NOISE_GATES + ["I"]) >= set(ops)
    assert len(ops) > 0

    for name in SQ_MEASUREMENTS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_MEASUREMENTS[name] in ops
        assert set(NOISE_GATES + [SQ_MEASUREMENTS[name]]) >= set(ops)
        # noise before the measurement
        assert ops.index(SQ_MEASUREMENTS[name]) == len(ops) - 1
        assert len(ops) > 1

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert ops == [TQ_GATES[name]]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert ops == ["DEPOLARIZE1"]

    return


def test_MeasurementNoiseModel():
    setup = Setup(SETUP)
    model = MeasurementNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_NOISELESS_OPS = SQ_GATES | SQ_RESETS
    for name in SQ_NOISELESS_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert ops == [SQ_NOISELESS_OPS[name]]

    for name in SQ_MEASUREMENTS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_MEASUREMENTS[name] in ops
        assert set(NOISE_GATES + [SQ_MEASUREMENTS[name]]) >= set(ops)
        # noise before the measurement
        assert ops.index(SQ_MEASUREMENTS[name]) == len(ops) - 1
        assert len(ops) > 1

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert ops == [TQ_GATES[name]]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_IncomingNoiseModel():
    setup = Setup(SETUP)
    model = IncomingNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_OPS = SQ_GATES | SQ_RESETS | SQ_MEASUREMENTS
    for name in SQ_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert ops == [SQ_OPS[name]]

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert ops == [TQ_GATES[name]]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert ops == ["X_ERROR", "Z_ERROR"]

    return


def test_IncomingDepolNoiseModel():
    setup = Setup(SETUP)
    model = IncomingDepolNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_OPS = SQ_GATES | SQ_RESETS | SQ_MEASUREMENTS
    for name in SQ_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert ops == [SQ_OPS[name]]

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert ops == [TQ_GATES[name]]

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert ops == ["DEPOLARIZE1"]

    return


def test_T1T2NoiseModel():
    setup = Setup(SETUP)
    model = T1T2NoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_OPS = SQ_GATES | SQ_RESETS | SQ_MEASUREMENTS
    for name in SQ_OPS:
        if name == "idle":
            continue
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_OPS[name] in ops
        assert set(NOISE_GATES + [SQ_OPS[name]]) >= set(ops)
        assert len(ops) > 1

    ops = [o.name for o in model.idle(["D1"])]
    assert ops == ["I"]

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert TQ_GATES[name] in ops
        assert set(NOISE_GATES + [TQ_GATES[name]]) >= set(ops)
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
    circ += model.reset(["D1"])
    circ += model.idle(["D2"])
    circ += model.tick()
    noise_channels = [o for o in circ if o.name in NOISE_GATES]
    assert len(noise_channels) == 2

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

    for name in SQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_GATES[name] in ops
        assert set(NOISE_GATES + [SQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

    for name in SQ_RESETS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_RESETS[name] in ops
        assert set(NOISE_GATES + [SQ_RESETS[name]]) >= set(ops)
        # noise after the reset
        assert ops.index(SQ_RESETS[name]) == 0
        assert len(ops) > 1

    for name in SQ_MEASUREMENTS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_MEASUREMENTS[name] in ops
        assert set(NOISE_GATES + [SQ_MEASUREMENTS[name]]) >= set(ops)
        # noise before the measurement
        assert ops.index(SQ_MEASUREMENTS[name]) == len(ops) - 1
        assert len(ops) > 1

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert TQ_GATES[name] in ops
        assert set(NOISE_GATES + [TQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

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


def test_BiasedCircuitNoiseModel():
    setup = Setup(SETUP)
    model = BiasedCircuitNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    SQ_OPS = SQ_GATES | SQ_RESETS | SQ_MEASUREMENTS
    for name in SQ_OPS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_OPS[name] in ops
        assert set(NOISE_GATES + [SQ_OPS[name]]) >= set(ops)
        assert len(ops) > 1

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert TQ_GATES[name] in ops
        assert set(NOISE_GATES + [TQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_CircuitNoiseModel():
    setup = Setup(SETUP)
    model = CircuitNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    for name in SQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_GATES[name] in ops
        assert set(NOISE_GATES + [SQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

    for name in SQ_RESETS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_RESETS[name] in ops
        assert set(NOISE_GATES + [SQ_RESETS[name]]) >= set(ops)
        # noise after the reset
        assert ops.index(SQ_RESETS[name]) == 0
        assert len(ops) > 1

    for name in SQ_MEASUREMENTS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_MEASUREMENTS[name] in ops
        assert set(NOISE_GATES + [SQ_MEASUREMENTS[name]]) >= set(ops)
        # noise before the measurement
        assert ops.index(SQ_MEASUREMENTS[name]) == len(ops) - 1
        assert len(ops) > 1

    for name in TQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert TQ_GATES[name] in ops
        assert set(NOISE_GATES + [TQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_MovableQubitsCircuitNoiseModel():
    setup = Setup(SETUP)
    model = MovableQubitsCircuitNoiseModel(setup, qubit_inds={"D1": 0, "D2": 1})

    for name in SQ_GATES:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_GATES[name] in ops
        assert set(NOISE_GATES + [SQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

    for name in SQ_RESETS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_RESETS[name] in ops
        assert set(NOISE_GATES + [SQ_RESETS[name]]) >= set(ops)
        # noise after the reset
        assert ops.index(SQ_RESETS[name]) == 0
        assert len(ops) > 1

    for name in SQ_MEASUREMENTS:
        ops = [o.name for o in model.__getattribute__(name)(["D1"])]
        assert SQ_MEASUREMENTS[name] in ops
        assert set(NOISE_GATES + [SQ_MEASUREMENTS[name]]) >= set(ops)
        # noise before the measurement
        assert ops.index(SQ_MEASUREMENTS[name]) == len(ops) - 1
        assert len(ops) > 1

    for name in TQ_GATES:
        if name == "swap":
            continue
        ops = [o.name for o in model.__getattribute__(name)(["D1", "D2"])]
        assert TQ_GATES[name] in ops
        assert set(NOISE_GATES + [TQ_GATES[name]]) >= set(ops)
        assert len(ops) > 1

    ops = [o.name for o in model.swap(["D1", "D2"])]
    assert "SWAP" in ops
    assert set(NOISE_GATES + ["SWAP"]) >= set(ops)
    assert ("DEPOLARIZE1" in ops) and ("DEPOLARIZE2" not in ops)
    assert len(ops) > 1

    ops = [o.name for o in model.incoming_noise(["D1"])]
    assert len(ops) == 0

    return


def test_model_meas_order():
    setup = Setup(SETUP)
    qubit_inds = {"D1": 1, "D2": 2}
    models = [
        CircuitNoiseModel(setup, qubit_inds=qubit_inds),
        NoiselessModel(qubit_inds=qubit_inds),
        T1T2NoiseModel(setup, qubit_inds=qubit_inds),
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

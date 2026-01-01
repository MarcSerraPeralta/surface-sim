from surface_sim import Model, Setup

SETUP = {
    "gate_durations": {
        "X": 3.2,
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
            "T1": 2.3,
            "T2": 1,
        },
    ],
}


def test_qubit_inds():
    setup = Setup(SETUP)
    qubit_inds = {"D1": 300, "d3": 2}
    model = Model(qubit_inds, setup)

    assert set(model.qubits) == set(qubit_inds.keys())

    for qubit, ind in qubit_inds.items():
        assert ind == model.get_inds([qubit])[0]

    return


def test_setup_from_model():
    setup = Setup(SETUP)
    qubit_inds = {"D1": 300, "d3": 2}
    model = Model(qubit_inds, setup)

    assert setup == model.setup
    assert SETUP["gate_durations"]["X"] == model.gate_duration("X")
    assert SETUP["setup"][0]["T1"] == model.param("T1")

    return


def test_new_circuit():
    setup = Setup(SETUP)
    qubit_inds = {"D1": 300, "d3": 2}
    model = Model(qubit_inds, setup)

    model._num_meas = 123
    model.new_circuit()

    assert model._num_meas == 0

    return

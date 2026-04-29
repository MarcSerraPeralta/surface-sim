import pytest

from surface_sim import Setup
from surface_sim.setups.random import uniform

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
            "sq_error_prob": "{free}",
            "tq_error_prob": 0.33,
            "cz_error_prob": 0.1,
            "meas_error_prob": "{free2}",
            "assign_error_flag": True,
            "assign_error_prob": 0.1,
            "reset_error_prob": 0.1,
            "idle_error_prob": 0.1,
            "T1": 2.3,
            "T2": 1,
        },
        {
            "qubit": "D1234",
            "sq_error_prob": 0.1234,
            "meas_error_prob": "{free2}",
            "hadamard_error_prob": "{free}",
        },
    ],
    "name": "test",
    "description": "test description",
}


def test_free_params():
    setup = Setup(SETUP)
    assert set(setup.free_params) == set(["free", "free2"])

    setup.set_var_param("free", 0.12)
    assert setup.param("sq_error_prob", "D1") == 0.12
    return


def test_unspecified_param():
    setup = Setup(SETUP)
    assert len(setup.free_params) > 0

    with pytest.raises(Exception):
        setup.param("sq_error_prob", "D1")
    return


def test_to_dict():
    setup = Setup(SETUP)
    setup_dict = setup.to_dict()
    assert setup_dict == SETUP
    return


def test_gate_duration():
    setup = Setup(SETUP)
    assert setup.gate_duration("X") == 3.2
    return


def test_parents():
    setup = Setup(SETUP)
    assert "swap_error_prob" not in SETUP["setup"]
    assert setup.param("swap_error_prob", "D1") == 0.33
    return


def test_random():
    setup = Setup(SETUP)

    with pytest.raises(ValueError):
        _ = setup.param("sq_error_prob", "D1")

    setup.convert_to_random(free=uniform(1, 2), free2=uniform(3, 4))

    assert setup.name == "test"
    assert setup.param("sq_error_prob", "D1") is not None
    assert setup.param("sq_error_prob", "D1234") == 0.1234

    setup = Setup(SETUP)
    setup.set_var_param("free", 0.123)

    setup.convert_to_random(free2=uniform(3, 4))

    assert setup.param("idle_error_prob", "D1") == 0.1
    assert setup.param("sq_error_prob", "D1") == 0.123
    assert setup.param("meas_error_prob", "D1") >= 3

    setup = Setup(SETUP)
    setup.set_var_param("free", 0.2)
    setup.set_var_param("free2", 0.3)

    setup.convert_to_random()

    assert setup.param("idle_error_prob", "D1") == 0.1
    assert setup.param("sq_error_prob", "D1") == 0.2
    assert setup.param("meas_error_prob", "D1") == 0.3

    return

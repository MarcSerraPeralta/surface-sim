import pytest

from surface_sim import Setup

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
            "sq_error_prob": "free",
            "cz_error_prob": 0.1,
            "meas_error_prob": "free2",
            "assign_error_flag": True,
            "assign_error_prob": 0.1,
            "reset_error_prob": 0.1,
            "idle_error_prob": 0.1,
            "T1": 2.3,
            "T2": 1,
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

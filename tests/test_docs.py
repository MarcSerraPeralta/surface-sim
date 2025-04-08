from surface_sim import Setup


def test_setups_in_docs():
    setup = Setup.from_yaml("./docs/setup_examples/biased_circ_level_noise.yaml")
    setup.set_var_param("prob", 0.1)
    setup.set_var_param("pauli", "X")
    assert setup.param("biased_pauli", ("D1",)) == "X"
    assert setup.param("sq_error_prob", ("D1",)) == 0.1

    setup = Setup.from_yaml("./docs/setup_examples/circ_level_noise.yaml")
    setup.set_var_param("prob", 0.1)
    assert setup.param("sq_error_prob", ("D1",)) == 0.1

    setup = Setup.from_yaml("./docs/setup_examples/device_specific_noise.yaml")
    assert setup.param("sq_error_prob", ("D1",)) == 0.0005

    return

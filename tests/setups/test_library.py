from surface_sim.setups import CircuitNoiseSetup, SI1000, SD6


def test_CircuitNoiseSetup():
    setup = CircuitNoiseSetup()
    setup.set_var_param("prob", 0.01)

    assert setup.param("sq_error_prob", ("D1",)) == 0.01

    return


def test_SD6():
    setup = SD6()
    setup.set_var_param("prob", 0.01)

    assert setup.param("sq_error_prob", ("D1",)) == 0.01

    return


def test_SI1000():
    setup = SI1000()
    setup.set_var_param("prob", 0.01)

    assert setup.param("sq_error_prob", ("D1",)) == 0.01 / 10

    return

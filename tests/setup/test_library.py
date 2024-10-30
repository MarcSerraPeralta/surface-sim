from surface_sim.setup import CircuitNoiseSetup


def test_CircuitNoiseSetup():
    setup = CircuitNoiseSetup()
    setup.set_var_param("prob", 0.01)
    return

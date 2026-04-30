from surface_sim.setups.random import (
    RandomSetupDict,
    gamma,
    lognormal,
    normal,
    uniform,
    weibull,
)


def test_normal():
    sample = normal(0, 1)()
    assert sample > -5 and sample < 5
    return


def test_gamma():
    sample = gamma(9, 0.5)()
    assert sample >= 0 and sample < 20
    return


def test_lognormal():
    sample = lognormal(0, 1)()
    assert sample > -5 and sample < 5
    return


def test_uniform():
    sample = uniform(0, 1)()
    assert sample >= 0 and sample <= 1
    return


def test_weibull():
    sample = weibull(1, 5)()
    assert sample >= 0 and sample <= 3
    return


def test_random_seed():
    for dist in [normal, gamma, lognormal, uniform, weibull]:
        first_sample = dist(1, 2, seed=123)()
        second_sample = dist(1, 2, seed=123)()
        assert first_sample == second_sample
    return


def test_RandomSetupDict():
    setup = RandomSetupDict({"name": "example", ("D1",): {"p": 10}})
    dist = uniform(0, 1)
    setup.sq_noise_sampler = lambda: {"p": 1}
    setup.tq_noise_sampler = lambda: {"p": dist()}
    assert setup["name"] == "example"
    assert setup[("D1",)] == {"p": 10}
    assert setup[("D2",)] == {"p": 1}
    assert setup[("D2",)] == {"p": 1}

    new_param = setup[("D1", "D2")]
    assert setup[("D1", "D2")] == new_param
    assert setup[("D2", "D3")] != new_param

    return

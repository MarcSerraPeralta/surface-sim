import random
from collections.abc import Callable

Param = float | int | bool | str | None


class RandomSetupDict(dict):
    """For each qubit or qubit pair, if the noise parameters are not defined,
    it samples a random physical error configuration given by the samplers
    and stores it in the dictionary.

    For reproducible results, check the implementation of each sampler.
    The implemented samplers in ``surface-sim`` have the argument ``seed``
    to specify the random seed.
    """

    def __init__(self, *args: object, **kargs: object):
        super().__init__(*args, **kargs)
        self.sq_noise_sampler: Callable[[], dict[str, Param]]
        self.tq_noise_sampler: Callable[[], dict[str, Param]]
        return

    def __missing__(self, qubits: tuple[str, ...]) -> dict[str, Param]:
        if not isinstance(qubits, tuple):
            raise TypeError(f"'qubits' must be a tuple, but {type(qubits)} was given.")
        if len(qubits) == 1:
            value = self.sq_noise_sampler()
        elif len(qubits) == 2:
            value = self.tq_noise_sampler()
        else:
            raise ValueError(
                f"Noise sampler for {len(qubits)}-qubit operation not implemented."
            )

        self[qubits] = value
        return value


def normal(mu: float, sigma: float, seed: int | None = None) -> Callable[[], float]:
    rng = random.Random(seed)

    def sampler() -> float:
        return rng.gauss(mu, sigma)

    return sampler


def lognormal(mu: float, sigma: float, seed: int | None = None) -> Callable[[], float]:
    rng = random.Random(seed)

    def sampler() -> float:
        return rng.lognormvariate(mu, sigma)

    return sampler


def gamma(alpha: float, beta: float, seed: int | None = None) -> Callable[[], float]:
    rng = random.Random(seed)

    def sampler() -> float:
        return rng.gammavariate(alpha, beta)

    return sampler


def weibull(alpha: float, beta: float, seed: int | None = None) -> Callable[[], float]:
    rng = random.Random(seed)

    def sampler() -> float:
        return rng.weibullvariate(alpha, beta)

    return sampler


def uniform(a: float, b: float, seed: int | None = None) -> Callable[[], float]:
    rng = random.Random(seed)

    def sampler() -> float:
        return rng.uniform(a, b)

    return sampler


def get_random_params(
    setup_dict: dict[str, list[dict[str, Param]]],
) -> dict[str, list[Param]]:
    """Returns the a dictionary mapping the error mechanism names to all
    the values of the given parameter in the setup.

    Parameters
    ----------
    setup_dict
        Dictionary version of the setup, which can be obtained by ``Setup.to_dict()``.
    """
    params: dict[str, list[Param]] = {}
    for _params in setup_dict["setup"]:
        if "qubit" in _params:
            _params.pop("qubit")
        elif "qubits" in _params:
            _params.pop("qubits")
        else:
            continue

        for param, value in _params.items():
            if param not in params:
                params[param] = []
            params[param].append(value)

    return params

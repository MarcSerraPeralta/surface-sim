from typing import Iterable, Iterator, Sequence, Tuple, List

from itertools import product

import numpy as np
from math import exp


def num_biased_ops(n):
    inds = np.arange(n)
    res = np.sum(np.power(4, inds) * np.power(3, n - 1 - inds))
    return res


def grouper(iterable: Iterable[str], block_size: int) -> Iterator[Tuple[str, ...]]:
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF ValueError
    args = [iter(iterable)] * block_size
    return zip(*args, strict=True)


def biased_prefactors(biased_pauli: str, biased_factor: float, num_qubits: int):
    """
    biased_prefactors Return a biased channel prefactors.

    The bias of the channel is defined as any error operator that
    applied the biased Pauli operator on any qubit.

    Parameters
    ----------
    biased_pauli : str
        The biased Pauli operator, represented as a string
    biased_factor : float
        The strength of the bias.

        A bias factor of 1 corresponds to a standard depolarizing channel.
        A bias factor of 0 leads to no probability of biased errors occurring.
        A bias channel tending towards infinify (but inf not supported) leads to
        only the biased errors occurring.
    num_qubits : int
        The number of qubits in the channel.

    Returns
    -------
    np.ndarray
        The array of prefactors
    """
    paulis = ["I", "X", "Y", "Z"]
    # get all pauli combinations and remove identity operator
    operators = list(product(paulis, repeat=num_qubits))[1:]
    num_ops = len(operators)

    biased_ops = [op for op in operators if biased_pauli in op]
    num_biased = len(biased_ops)

    nonbias_prefactor = 1 / (num_biased * (biased_factor - 1) + num_ops)
    bias_prefactor = biased_factor * nonbias_prefactor

    prefactors = []
    for op in operators:
        if biased_pauli in op:
            prefactors.append(bias_prefactor)
        else:
            prefactors.append(nonbias_prefactor)
    prefactors = np.array(prefactors)

    return prefactors


def idle_error_probs(
    relax_time: float, deph_time: float, duration: float
) -> List[float]:
    """
    idle_error_probs Returns the probabilities of X, Y, and Z errors
    for a Pauli-twirled amplitude and phase damping channel.

    References:
    arXiv:1210.5799
    arXiv:1305.2021

    Parameters
    ----------
    relax_time : float
        The relaxation time (T1) of the qubit.
    deph_time : float
        The dephasing time (T2) of the qubit.
    duration : float
        The duration of the amplitude-phase damping period.

    Returns
    -------
    List[float, float, float]
        The probabilities of X, Y, and Z errors
    """
    # Check for invalid inputs
    # If either the duration, relaxation time, or dephasing time is negative, you get negative probabilities
    # If the relaxation time or dephasing time is zero, you get a divide by zero error.
    # If the duration is zero, you don't get any errors, but it is likely a bug in the code for the user to have a zero duration.
    if relax_time <= 0:
        raise ValueError("The relaxation time ('relax_time') must be positive")
    if deph_time <= 0:
        raise ValueError("The dephasing time ('deph_time') must be positive")
    if duration <= 0:
        raise ValueError("The idling duration ('duration') must be positive")

    relax_prob = 1 - exp(-duration / relax_time)
    deph_prob = 1 - exp(-duration / deph_time)

    x_prob = y_prob = 0.25 * relax_prob
    z_prob = 0.5 * deph_prob - 0.25 * relax_prob

    return [x_prob, y_prob, z_prob]

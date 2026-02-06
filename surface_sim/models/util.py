from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from itertools import product
from math import dist, exp
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from stim import Circuit, CircuitInstruction

from ..setups.setup import (
    LONG_RANGE_TQ_GATES,
    SQ_GATES,
    SQ_MEASUREMENTS,
    SQ_RESETS,
    TQ_GATES,
)
from .model import Model

T = TypeVar("T")
ALL_TQ_GATES = TQ_GATES | LONG_RANGE_TQ_GATES


def num_biased_ops(n: int) -> int:
    inds = np.arange(n)
    return np.sum(np.power(4, inds) * np.power(3, n - 1 - inds))


def grouper(iterable: Iterable[T], block_size: int) -> Iterator[tuple[T, ...]]:
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF ValueError
    args = [iter(iterable)] * block_size
    return zip(*args, strict=True)


def biased_prefactors(
    biased_pauli: str, biased_factor: float, num_qubits: int
) -> npt.NDArray[np.float64]:
    """Return a biased channel prefactors.

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
    prefactors
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

    prefactors: list[float] = []
    for op in operators:
        if biased_pauli in op:
            prefactors.append(bias_prefactor)
        else:
            prefactors.append(nonbias_prefactor)
    prefactors_np = np.array(prefactors)

    return prefactors_np


def idle_error_probs(
    relax_time: float, deph_time: float, duration: float
) -> tuple[float, float, float]:
    """Returns the probabilities of X, Y, and Z errors
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
    tuple[float, float, float]
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

    return (x_prob, y_prob, z_prob)


def sq_gate_depol1_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_gate(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction(SQ_GATES[name], inds))
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(f"{name}_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    return sq_gate


def tq_gate_depol2_generator(
    self: Model, name: str
) -> Callable[[Sequence[str]], Circuit]:
    def tq_gate(qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction(ALL_TQ_GATES[name], inds))
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE2", inds, [prob]))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob: float = self.param(f"{name}_error_prob", qubit_pair)
                circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    return tq_gate


def tq_gate_depol1_generator(
    self: Model, name: str
) -> Callable[[Sequence[str]], Circuit]:
    def tq_gate(qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction(ALL_TQ_GATES[name], inds))
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob: float = self.param(f"{name}_error_prob", qubit_pair)
                circ.append(CircuitInstruction("DEPOLARIZE1", ind_pair, [prob]))
        return circ

    return tq_gate


def long_range_tq_gate_depol2_generator(
    self: Model, name: str
) -> Callable[[Sequence[str]], Circuit]:
    def cphase(qubits: Sequence[str]) -> Circuit:
        circuit = Circuit()
        for pair in grouper(qubits, 2):
            distance = dist(self._qubit_coords[pair[0]], self._qubit_coords[pair[1]])
            _name = name
            if distance > self.setup.param("long_coupler_distance"):
                _name = f"long_range_{name}"
            attr = tq_gate_depol2_generator(self, _name)
            circuit += attr(pair)
        return circuit

    return cphase


def sq_meas_x_assign_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_meas(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        noise_name = "X_ERROR" if "_x" not in name else "Z_ERROR"
        circ = Circuit()

        # separates X_ERROR and MZ lines for clearer stim.Circuits and diagrams
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            circ.append(CircuitInstruction(noise_name, inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob: float = self.param("assign_error_prob")
                circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], inds, [prob]))
            else:
                circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], inds))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(f"{name}_error_prob", qubit)
                circ.append(CircuitInstruction(noise_name, [ind], [prob]))
            for qubit, ind in zip(qubits, inds):
                self.add_meas(qubit)
                if self.param("assign_error_flag", qubit):
                    prob: float = self.param("assign_error_prob", qubit)
                    circ.append(
                        CircuitInstruction(SQ_MEASUREMENTS[name], [ind], [prob])
                    )
                else:
                    circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], [ind]))

        return circ

    return sq_meas


def sq_reset_x_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_reset(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        noise_name = "X_ERROR" if "_x" not in name else "Z_ERROR"
        circ = Circuit()

        circ.append(CircuitInstruction(SQ_RESETS[name], inds))
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            circ.append(CircuitInstruction(noise_name, inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(f"{name}_error_prob", qubit)
                circ.append(CircuitInstruction(noise_name, [ind], [prob]))

        return circ

    return sq_reset


def sq_gate_biased_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_gate(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction(SQ_GATES[name], inds))
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, probs))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(f"{name}_error_prob", qubit)
                prefactors = biased_prefactors(
                    biased_pauli=self.param("biased_pauli", qubit),
                    biased_factor=self.param("biased_factor", qubit),
                    num_qubits=1,
                )
                probs = prob * prefactors
                circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], probs))
        return circ

    return sq_gate


def tq_gate_biased_generator(
    self: Model, name: str
) -> Callable[[Sequence[str]], Circuit]:
    def tq_gate(qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction(TQ_GATES[name], inds))
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", inds, probs))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob: float = self.param(f"{name}_error_prob", qubit_pair)
                prefactors = biased_prefactors(
                    biased_pauli=self.param("biased_pauli", qubit_pair),
                    biased_factor=self.param("biased_factor", qubit_pair),
                    num_qubits=2,
                )
                probs = prob * prefactors
                circ.append(CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs))
        return circ

    return tq_gate


def sq_gate_noiseless_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_gate(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        circ.append(CircuitInstruction(SQ_GATES[name], inds))
        return circ

    return sq_gate


def tq_gate_noiseless_generator(
    self: Model, name: str
) -> Callable[[Sequence[str]], Circuit]:
    def tq_gate(qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()
        circ.append(CircuitInstruction(TQ_GATES[name], inds))
        return circ

    return tq_gate


def sq_meas_noiseless_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_meas(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        for qubit in qubits:
            self.add_meas(qubit)
        circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], inds))
        return circ

    return sq_meas


def sq_reset_noiseless_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_reset(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        circ.append(CircuitInstruction(SQ_RESETS[name], inds))
        return circ

    return sq_reset


def sq_meas_depol1_assign_generator(
    self: Model, name: str
) -> Callable[[Collection[str]], Circuit]:
    def sq_meas(qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ lines for clearer stim.Circuits and diagrams
        if self.uniform:
            prob: float = self.param(f"{name}_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob: float = self.param("assign_error_prob")
                circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], inds, [prob]))
            else:
                circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], inds))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(f"{name}_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
            for qubit, ind in zip(qubits, inds):
                self.add_meas(qubit)
                if self.param("assign_error_flag", qubit):
                    prob: float = self.param("assign_error_prob", qubit)
                    circ.append(
                        CircuitInstruction(SQ_MEASUREMENTS[name], [ind], [prob])
                    )
                else:
                    circ.append(CircuitInstruction(SQ_MEASUREMENTS[name], [ind]))

        return circ

    return sq_meas

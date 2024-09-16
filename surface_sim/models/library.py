from typing import Iterator, Dict, Iterable

from stim import CircuitInstruction

from ..setup import Setup
from .model import Model
from .util import biased_prefactors, grouper, idle_error_probs


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup, qubit_inds: Dict[str, int]) -> None:
        super().__init__(setup, qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def z_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("Z", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def hadamard(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def s_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("S", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def s_dag_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("S_DAG", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def cphase(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            yield CircuitInstruction("DEPOLARIZE2", ind_pair, [prob])

    def measure(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)

            prob = self.param("meas_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                yield CircuitInstruction("MZ", [ind], [prob])
            else:
                yield CircuitInstruction("MZ", [ind])

    def reset(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

    def idle(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])


class BiasedCircuitNoiseModel(Model):
    def __init__(self, setup: Setup, qubit_inds: Dict[str, int]) -> None:
        super().__init__(setup, qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def z_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("Z", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def hadamard(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def s_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("S", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def s_dag_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("S_DAG", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def cphase(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", *qubit_pair),
                biased_factor=self.param("biased_factor", *qubit_pair),
                num_qubits=2,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs)

    def measure(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)

            prob = self.param("meas_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                yield CircuitInstruction("MZ", [ind], [prob])
            else:
                yield CircuitInstruction("MZ", [ind])

    def reset(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

    def idle(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            prob = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], prob)


class DecoherenceNoiseModel(Model):
    """An coherence-limited noise model using T1 and T2"""

    def __init__(
        self, setup: Setup, qubit_inds: Dict[str, int], symmetric_noise: bool = True
    ) -> None:
        self._sym_noise = symmetric_noise
        return super().__init__(setup, qubit_inds)

    def generic_op(
        self, name: str, qubits: Iterable[str]
    ) -> Iterator[CircuitInstruction]:
        """
        generic_op Returns the circuit instructions for a generic operation (that is supported by Stim) on the given qubits.

        Parameters
        ----------
        name
            The name of the gate (as defined in Stim)
        qubits
            The qubits to apply the gate to.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for a generic gate on the given qubits.
        """
        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            yield from self.idle(qubits, duration)
            yield CircuitInstruction(name, targets=self.get_inds(qubits))
            yield from self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            yield CircuitInstruction(name, targets=self.get_inds(qubits))
            yield from self.idle(qubits, duration)

    def x_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("X", qubits)

    def z_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("Z", qubits)

    def hadamard(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("H", qubits)

    def s_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("S", qubits)

    def s_dag_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("S_DAG", qubits)

    def cphase(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("CZ", qubits)

    def measure(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        name = "M"
        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            yield from self.idle(qubits, duration)
            for qubit in qubits:
                self.add_meas(qubit)

                if self.param("assign_error_flag", qubit):
                    prob = self.param("assign_error_prob", qubit)
                    yield CircuitInstruction(
                        name, targets=self.get_inds([qubit]), gate_args=[prob]
                    )
                else:
                    yield CircuitInstruction(name, targets=self.get_inds([qubit]))
            yield from self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            for qubit in qubits:
                self.add_meas(qubit)

                yield CircuitInstruction(name, targets=[qubit])
                yield from self.idle(qubit, duration)

    def reset(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("R", qubits)

    def idle(
        self, qubits: Iterable[str], duration: float
    ) -> Iterator[CircuitInstruction]:
        """
        idle Returns the circuit instructions for an idling period on the given qubits.

        Parameters
        ----------
        qubits
            The qubits to idle.
        duration
            The duration of the idling period.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an idling period on the given qubits.
        """
        for qubit in qubits:
            relax_time = self.param("T1", qubit)
            deph_time = self.param("T2", qubit)
            # check that the parameters are physical
            assert (relax_time > 0) and (deph_time > 0) and (deph_time < 2 * relax_time)

            error_probs = idle_error_probs(relax_time, deph_time, duration)

            yield CircuitInstruction(
                "PAULI_CHANNEL_1", targets=self.get_inds([qubit]), gate_args=error_probs
            )


class ExperimentalNoiseModel(Model):
    """
    Noise models that uses the metrics characterized from
    an experimental setup
    """

    def x_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def hadamard(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def cphase(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            yield CircuitInstruction("DEPOLARIZE2", ind_pair, [prob])

    def measure(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)

            prob = self.param("meas_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                yield CircuitInstruction("MZ", [ind], [prob])
            else:
                yield CircuitInstruction("MZ", [ind])

    def reset(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

    def idle(
        self, qubits: Iterable[str], duration: float
    ) -> Iterator[CircuitInstruction]:
        """
        idle Returns the circuit instructions for an idling period on the given qubits.

        Parameters
        ----------
        qubits
            The qubits to idle.
        duration
            The duration of the idling period.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an idling period on the given qubits.
        """
        for qubit in qubits:
            relax_time = self.param("T1", qubit)
            deph_time = self.param("T2", qubit)
            # check that the parameters are physical
            assert (relax_time > 0) and (deph_time > 0) and (deph_time < 2 * relax_time)

            error_probs = idle_error_probs(relax_time, deph_time, duration)

            yield CircuitInstruction(
                "PAULI_CHANNEL_1", targets=self.get_inds([qubit]), gate_args=error_probs
            )


class NoiselessModel(Model):
    """Noiseless model"""

    def __init__(self, qubit_inds: Dict[str, int]) -> None:
        empty_setup = Setup(dict(setup=[{}]))
        return super().__init__(setup=empty_setup, qubit_inds=qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("X", self.get_inds(qubits))

    def z_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("Z", self.get_inds(qubits))

    def hadamard(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("H", self.get_inds(qubits))

    def s_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("S", self.get_inds(qubits))

    def s_dag_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("S_DAG", self.get_inds(qubits))

    def cphase(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("CZ", self.get_inds(qubits))

    def measure(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        for qubit in qubits:
            self.add_meas(qubit)
            yield CircuitInstruction("M", self.get_inds([qubit]))

    def reset(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("R", self.get_inds(qubits))

    def idle(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("I", self.get_inds(qubits))

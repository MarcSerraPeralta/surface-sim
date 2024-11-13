from collections.abc import Iterable, Sequence

from stim import CircuitInstruction, Circuit

from ..setup import Setup
from .model import Model
from .util import biased_prefactors, grouper, idle_error_probs


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        super().__init__(setup, qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("X", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("x_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("Z", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("z_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("H", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("h_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("S", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("s_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("S_DAG", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sdag_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def cphase(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CZ", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def cnot(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CNOT", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cnot_error_prob", *qubit_pair)
            circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def swap(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("SWAP", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("swap_error_prob", *qubit_pair)
            circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MZ", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MZ", [ind]))

        return circ

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MX", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MX", [ind]))

        return circ

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MY", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MY", [ind]))

        return circ

    def reset(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("R", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))
        return circ

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RX", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RY", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def idle(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class BiasedCircuitNoiseModel(Model):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        super().__init__(setup, qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("X", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("x_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], probs))
        return circ

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("Z", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("z_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], probs))
        return circ

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("H", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("h_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], probs))
        return circ

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("S", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("s_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], probs))
        return circ

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("S_DAG", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sdag_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], probs))
        return circ

    def cphase(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CZ", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", *qubit_pair),
                biased_factor=self.param("biased_factor", *qubit_pair),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs))
        return circ

    def cnot(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CNOT", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cnot_error_prob", *qubit_pair)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", *qubit_pair),
                biased_factor=self.param("biased_factor", *qubit_pair),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs))
        return circ

    def swap(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("SWAP", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("swap_error_prob", *qubit_pair)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", *qubit_pair),
                biased_factor=self.param("biased_factor", *qubit_pair),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs))
        return circ

    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MZ", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MZ", [ind]))

        return circ

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MX", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MX", [ind]))

        return circ

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MX", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MX", [ind]))

        return circ

    def reset(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("R", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))
        return circ

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RX", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RY", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def idle(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            prob = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], prob))
        return circ

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class DecoherenceNoiseModel(Model):
    """An coherence-limited noise model using T1 and T2"""

    def __init__(
        self, setup: Setup, qubit_inds: dict[str, int], symmetric_noise: bool = True
    ) -> None:
        self._sym_noise = symmetric_noise
        return super().__init__(setup, qubit_inds)

    def generic_op(self, name: str, qubits: Iterable[str]) -> Circuit:
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
        Circuit
            The circuit instructions for a generic gate on the given qubits.
        """
        circ = Circuit()

        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            circ += self.idle(qubits, duration)
            circ.append(CircuitInstruction(name, targets=self.get_inds(qubits)))
            circ += self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            circ.append(CircuitInstruction(name, targets=self.get_inds(qubits)))
            circ += self.idle(qubits, duration)
        return circ

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("X", qubits)

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("Z", qubits)

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("H", qubits)

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("S", qubits)

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("S_DAG", qubits)

    def cphase(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("CZ", qubits)

    def cnot(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("CNOT", qubits)

    def swap(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("SWAP", qubits)

    def measure(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        name = "M"
        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            circ += self.idle(qubits, duration)
            for qubit in qubits:
                self.add_meas(qubit)

                if self.param("assign_error_flag", qubit):
                    prob = self.param("assign_error_prob", qubit)
                    circ.append(
                        CircuitInstruction(
                            name, targets=self.get_inds([qubit]), gate_args=[prob]
                        )
                    )
                else:
                    circ.append(
                        CircuitInstruction(name, targets=self.get_inds([qubit]))
                    )
            circ += self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            for qubit in qubits:
                self.add_meas(qubit)

                circ.append(CircuitInstruction(name, targets=[qubit]))
                circ += self.idle(qubit, duration)
        return circ

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        name = "MX"
        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            circ += self.idle(qubits, duration)
            for qubit in qubits:
                self.add_meas(qubit)

                if self.param("assign_error_flag", qubit):
                    prob = self.param("assign_error_prob", qubit)
                    circ.append(
                        CircuitInstruction(
                            name, targets=self.get_inds([qubit]), gate_args=[prob]
                        )
                    )
                else:
                    circ.append(
                        CircuitInstruction(name, targets=self.get_inds([qubit]))
                    )
            circ += self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            for qubit in qubits:
                self.add_meas(qubit)

                circ.append(CircuitInstruction(name, targets=[qubit]))
                circ += self.idle(qubit, duration)
        return circ

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        name = "MY"
        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            circ += self.idle(qubits, duration)
            for qubit in qubits:
                self.add_meas(qubit)

                if self.param("assign_error_flag", qubit):
                    prob = self.param("assign_error_prob", qubit)
                    circ.append(
                        CircuitInstruction(
                            name, targets=self.get_inds([qubit]), gate_args=[prob]
                        )
                    )
                else:
                    circ.append(
                        CircuitInstruction(name, targets=self.get_inds([qubit]))
                    )
            circ += self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            for qubit in qubits:
                self.add_meas(qubit)

                circ.append(CircuitInstruction(name, targets=[qubit]))
                circ += self.idle(qubit, duration)
        return circ

    def reset(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("R", qubits)

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("RX", qubits)

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        return self.generic_op("RY", qubits)

    def idle(self, qubits: Iterable[str], duration: float) -> Circuit:
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
        Circuit
            The circuit instructions for an idling period on the given qubits.
        """
        circ = Circuit()

        for qubit in qubits:
            relax_time = self.param("T1", qubit)
            deph_time = self.param("T2", qubit)
            # check that the parameters are physical
            assert (relax_time > 0) and (deph_time > 0) and (deph_time < 2 * relax_time)

            error_probs = idle_error_probs(relax_time, deph_time, duration)

            circ.append(
                CircuitInstruction(
                    "PAULI_CHANNEL_1",
                    targets=self.get_inds([qubit]),
                    gate_args=error_probs,
                )
            )
        return circ

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class ExperimentalNoiseModel(Model):
    """
    Noise models that uses the metrics characterized from
    an experimental setup
    """

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("X", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("x_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))

        return circ

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("Z", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("z_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))

        return circ

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("H", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("h_error_prob", qubit)
            circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def cphase(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CZ", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def cnot(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CNOT", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cnot_error_prob", *qubit_pair)
            circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def swap(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("SWAP", inds))

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("swap_error_prob", *qubit_pair)
            circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MZ", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MZ", [ind]))

        return circ

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MX", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MX", [ind]))

        return circ

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MY", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MY", [ind]))

        return circ

    def reset(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("R", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))
        return circ

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RX", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RY", inds))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))
        return circ

    def idle(self, qubits: Iterable[str], duration: float) -> Circuit:
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
        Circuit
            The circuit instructions for an idling period on the given qubits.
        """
        circ = Circuit()
        for qubit in qubits:
            relax_time = self.param("T1", qubit)
            deph_time = self.param("T2", qubit)
            # check that the parameters are physical
            assert (relax_time > 0) and (deph_time > 0) and (deph_time < 2 * relax_time)

            error_probs = idle_error_probs(relax_time, deph_time, duration)

            circ.append(
                CircuitInstruction(
                    "PAULI_CHANNEL_1",
                    targets=self.get_inds([qubit]),
                    gate_args=error_probs,
                )
            )
        return circ

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class NoiselessModel(Model):
    """Noiseless model"""

    def __init__(self, qubit_inds: dict[str, int]) -> None:
        empty_setup = Setup(dict(setup=[{}]))
        return super().__init__(setup=empty_setup, qubit_inds=qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("X", self.get_inds(qubits)))
        return circ

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("Z", self.get_inds(qubits)))
        return circ

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("H", self.get_inds(qubits)))
        return circ

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("S", self.get_inds(qubits)))
        return circ

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("S_DAG", self.get_inds(qubits)))
        return circ

    def cphase(self, qubits: Sequence[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("CZ", self.get_inds(qubits)))
        return circ

    def cnot(self, qubits: Sequence[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("CNOT", self.get_inds(qubits)))
        return circ

    def swap(self, qubits: Sequence[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("SWAP", self.get_inds(qubits)))
        return circ

    def measure(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        for qubit in qubits:
            self.add_meas(qubit)
            circ.append(CircuitInstruction("M", self.get_inds([qubit])))
        return circ

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        for qubit in qubits:
            self.add_meas(qubit)
            circ.append(CircuitInstruction("MX", self.get_inds([qubit])))
        return circ

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        for qubit in qubits:
            self.add_meas(qubit)
            circ.append(CircuitInstruction("MY", self.get_inds([qubit])))
        return circ

    def reset(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("R", self.get_inds(qubits)))
        return circ

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("RX", self.get_inds(qubits)))
        return circ

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("RY", self.get_inds(qubits)))
        return circ

    def idle(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("I", self.get_inds(qubits)))
        return circ

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class IncomingNoiseModel(NoiselessModel):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        self._setup = setup
        self._qubit_inds = qubit_inds
        self._meas_order = {q: [] for q in qubit_inds}
        self._num_meas = 0
        return

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # Split the 'for' loop in two so that the stim diagram looks better
        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        return circ


class PhenomenologicalNoiseModel(IncomingNoiseModel):
    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MZ", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MZ", [ind]))

        return circ

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MX", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MX", [ind]))

        return circ

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

        for qubit, ind in zip(qubits, inds):
            self.add_meas(qubit)
            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                circ.append(CircuitInstruction("MY", [ind], [prob]))
            else:
                circ.append(CircuitInstruction("MY", [ind]))

        return circ

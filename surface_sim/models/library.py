from __future__ import annotations
from collections.abc import Iterable, Sequence

from stim import CircuitInstruction, Circuit

from ..setup import Setup
from ..layouts import Layout
from .model import Model
from .util import biased_prefactors, grouper, idle_error_probs


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        super().__init__(setup, qubit_inds)

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("X", inds))
        if self.uniform:
            prob = self.param("x_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("x_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("Z", inds))
        if self.uniform:
            prob = self.param("z_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("z_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("H", inds))
        if self.uniform:
            prob = self.param("h_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("h_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("S", inds))
        if self.uniform:
            prob = self.param("s_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("s_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("S_DAG", inds))
        if self.uniform:
            prob = self.param("sdag_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
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
        if self.uniform:
            prob = self.param("cz_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE2", inds, [prob]))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob = self.param("cz_error_prob", qubit_pair)
                circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def cnot(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("CNOT", inds))
        if self.uniform:
            prob = self.param("cnot_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE2", inds, [prob]))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob = self.param("cnot_error_prob", qubit_pair)
                circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def swap(self, qubits: Sequence[str]) -> Circuit:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("SWAP", inds))
        if self.uniform:
            prob = self.param("swap_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE2", inds, [prob]))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob = self.param("swap_error_prob", qubit_pair)
                circ.append(CircuitInstruction("DEPOLARIZE2", ind_pair, [prob]))
        return circ

    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MZ", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MZ", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MX", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MX", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MY", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MY", inds))
        else:
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
        if self.uniform:
            prob = self.param("reset_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("reset_error_prob", qubit)
                circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))
        return circ

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RX", inds))
        if self.uniform:
            prob = self.param("reset_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("reset_error_prob", qubit)
                circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RY", inds))
        if self.uniform:
            prob = self.param("reset_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("reset_error_prob", qubit)
                circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def idle(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits)

        return circ

    def idle_meas(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits, "idle_meas_error_prob")

        return circ

    def idle_reset(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits, "idle_reset_error_prob")

        return circ

    def idle_noise(
        self, qubits: Iterable[str], param_name: str = "idle_error_prob"
    ) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        if self.uniform:
            prob = self.param(param_name)
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param(param_name, qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class SI1000NoiseModel(CircuitNoiseModel):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        self._meas_or_reset_qubits = []
        super().__init__(setup, qubit_inds)
        return

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        meas_reset_ops = [
            "measure",
            "measure_x",
            "measure_y",
            "measure_z",
            "reset",
            "reset_x",
            "reset_y",
            "reset_z",
        ]
        if callable(attr) and (name in meas_reset_ops):

            def wrapper(qubits: Iterable[str], *args, **kargs):
                self._meas_or_reset_qubits += list(qubits)
                return attr(qubits, *args, **kargs)

            return wrapper
        return attr

    def flush_noise(self) -> Circuit:
        circ = Circuit()
        if self._meas_or_reset_qubits:
            idle_qubits = set(self._qubit_inds).difference(self._meas_or_reset_qubits)
            circ += self.idle_noise(idle_qubits, "extra_idle_meas_or_reset_error_prob")
        self._meas_or_reset_qubits = []
        return circ


class BiasedCircuitNoiseModel(Model):
    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("X", inds))

        if self.uniform:
            prob = self.param("x_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, probs))
        else:
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

        if self.uniform:
            prob = self.param("z_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, probs))
        else:
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

        if self.uniform:
            prob = self.param("h_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, probs))
        else:
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

        if self.uniform:
            prob = self.param("s_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, probs))
        else:
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

        if self.uniform:
            prob = self.param("sdag_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, probs))
        else:
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

        if self.uniform:
            prob = self.param("cz_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", inds, probs))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob = self.param("cz_error_prob", qubit_pair)
                prefactors = biased_prefactors(
                    biased_pauli=self.param("biased_pauli", qubit_pair),
                    biased_factor=self.param("biased_factor", qubit_pair),
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

        if self.uniform:
            prob = self.param("cnot_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", inds, probs))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob = self.param("cnot_error_prob", qubit_pair)
                prefactors = biased_prefactors(
                    biased_pauli=self.param("biased_pauli", qubit_pair),
                    biased_factor=self.param("biased_factor", qubit_pair),
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

        if self.uniform:
            prob = self.param("swap_error_prob")
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=2,
            )
            probs = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_2", inds, probs))
        else:
            for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
                prob = self.param("swap_error_prob", qubit_pair)
                prefactors = biased_prefactors(
                    biased_pauli=self.param("biased_pauli", qubit_pair),
                    biased_factor=self.param("biased_factor", qubit_pair),
                    num_qubits=2,
                )
                probs = prob * prefactors
                circ.append(CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs))
        return circ

    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MZ", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MZ", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MX", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MX", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MY", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MY", inds))
        else:
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

        if self.uniform:
            prob = self.param("reset_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("reset_error_prob", qubit)
                circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))
        return circ

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RX", inds))

        if self.uniform:
            prob = self.param("reset_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("reset_error_prob", qubit)
                circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("RY", inds))

        if self.uniform:
            prob = self.param("reset_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("reset_error_prob", qubit)
                circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))
        return circ

    def idle(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits)

        return circ

    def idle_meas(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits, "idle_meas_error_prob")

        return circ

    def idle_reset(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits, "idle_reset_error_prob")

        return circ

    def idle_noise(
        self, qubits: Iterable[str], param_name: str = "idle_error_prob"
    ) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        if self.uniform:
            prob = self.param(param_name)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            prob = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, prob))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param(param_name, qubit)
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
    """A coherence-limited noise model using T1 and T2.
    The noise is added when perfoming gates and when calling
    ``DecoherenceNoiseModel.tick``.
    """

    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        self._durations = {q: 0.0 for q in qubit_inds}
        super().__init__(setup=setup, qubit_inds=qubit_inds)
        return

    def _generic_gate(self, name: str, qubits: Iterable[str]) -> Circuit:
        """
        Returns the circuit instructions for a generic gate supported by
        ``stim`` on the given qubits.

        Parameters
        ----------
        name
            The name of the gate as defined in ``stim``.
        qubits
            The qubits to apply the gate to.

        Returns
        -------
        circ
            The circuit instructions for a generic gate on the given qubits.
        """
        sym_noise = set(self.setup.param("symmetric_noise", q) for q in qubits)
        if len(sym_noise) != 1:
            raise ValueError(
                "'sym_noise' has different values for the considered qubits."
            )
        sym_noise = sym_noise.pop()

        circ = Circuit()
        duration = self.gate_duration(name)
        if sym_noise:
            duration = 0.5 * duration
            circ += self.idle_duration(qubits, duration)

        circ.append(CircuitInstruction(name, targets=self.get_inds(qubits)))
        circ += self.idle_duration(qubits, duration)

        return circ

    def _generic_measurement(self, name: str, qubits: Iterable[str]) -> Circuit:
        """
        Returns the circuit instructions for a generic measurement supported by
        ``stim`` on the given qubits.

        Parameters
        ----------
        name
            The name of the measurement as defined in ``stim``.
        qubits
            The qubits to apply the gate to.

        Returns
        -------
        circ
            The circuit instructions for a generic measurement on the given qubits.
        """
        sym_noise = set(self.setup.param("symmetric_noise", q) for q in qubits)
        if len(sym_noise) != 1:
            raise ValueError(
                "'sym_noise' has different values for the considered qubits."
            )
        sym_noise = sym_noise.pop()

        circ = Circuit()
        duration = self.gate_duration(name)
        if sym_noise:
            duration = 0.5 * duration
            circ += self.idle_duration(qubits, duration)

        for qubit in qubits:
            self.add_meas(qubit)
            inds = self.get_inds([qubit])
            if prob := self.param("assign_error_flag", qubit):
                circ.append(CircuitInstruction(name, targets=inds, gate_args=[prob]))
            else:
                circ.append(CircuitInstruction(name, targets=inds))

        circ += self.idle_duration(qubits, duration)

        return circ

    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("X")
        return self._generic_gate("X", qubits)

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("Z")
        return self._generic_gate("Z", qubits)

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("H")
        return self._generic_gate("H", qubits)

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("S")
        return self._generic_gate("S", qubits)

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("S_DAG")
        return self._generic_gate("S_DAG", qubits)

    def cphase(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("CZ")
        return self._generic_gate("CZ", qubits)

    def cnot(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("CNOT")
        return self._generic_gate("CNOT", qubits)

    def swap(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("SWAP")
        return self._generic_gate("SWAP", qubits)

    def measure(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("M")
        return self._generic_measurement("M", qubits)

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("MX")
        return self._generic_measurement("MX", qubits)

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("MY")
        return self._generic_measurement("MY", qubits)

    def reset(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("R")
        return self._generic_gate("R", qubits)

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("RX")
        return self._generic_gate("RX", qubits)

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        for qubit in qubits:
            self._durations[qubit] += self.gate_duration("RY")
        return self._generic_gate("RY", qubits)

    def idle(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        circ.append(CircuitInstruction("I", inds))
        return circ

    def idle_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()

    def idle_meas(self, qubits: Iterable[str]) -> Circuit:
        return self.idle(qubits)

    def idle_reset(self, qubits: Iterable[str]) -> Circuit:
        return self.idle(qubits)

    def flush_noise(self) -> Circuit:
        # compute idling time for each qubit
        max_duration = max(self._durations.values())
        durations = {q: max_duration - d for q, d in self._durations.items()}
        durations = {q: d for q, d in durations.items() if d != 0}

        # order durations for better circuit readibility
        durations = sorted(durations.items(), key=lambda x: x[1])

        # build circuit
        circ = Circuit()
        for qubit, duration in durations:
            circ += self.idle_duration([qubit], duration)

        # reset durations
        self._durations = {q: 0.0 for q in self._qubit_inds}

        return circ

    def idle_duration(self, qubits: Iterable[str], duration: float) -> Circuit:
        """Returns the circuit instructions for an idling period on the given qubits.

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

        if self.uniform:
            relax_time = self.param("T1")
            deph_time = self.param("T2")
            # check that the parameters are physical
            assert (relax_time > 0) and (deph_time > 0) and (deph_time < 2 * relax_time)

            error_probs = idle_error_probs(relax_time, deph_time, duration)
            targets = self.get_inds(qubits)

            circ.append(
                CircuitInstruction(
                    "PAULI_CHANNEL_1", targets=targets, gate_args=error_probs
                )
            )

            return circ

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

    def __init__(
        self, qubit_inds: dict[str, int], setup: Setup = Setup(dict(setup=[{}]))
    ) -> None:
        return super().__init__(setup=setup, qubit_inds=qubit_inds)

    @classmethod
    def from_layouts(cls: type[NoiselessModel], *layouts: Layout) -> "NoiselessModel":
        """Creates a ``Model`` object using the information from the layouts."""
        qubit_inds = {}
        for layout in layouts:
            qubit_inds |= layout.qubit_inds  # updates dict
        return cls(qubit_inds=qubit_inds)

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

    def idle_meas(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("I", self.get_inds(qubits)))
        return circ

    def idle_reset(self, qubits: Iterable[str]) -> Circuit:
        circ = Circuit()
        circ.append(CircuitInstruction("I", self.get_inds(qubits)))
        return circ

    def idle_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        return Circuit()


class IncomingNoiseModel(NoiselessModel):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        return Model.__init__(self, setup=setup, qubit_inds=qubit_inds)

    @classmethod
    def from_layouts(
        cls: type[IncomingNoiseModel], setup: Setup, *layouts: Layout
    ) -> "IncomingNoiseModel":
        """Creates a ``Model`` object using the information from the layouts."""
        qubit_inds = {}
        for layout in layouts:
            qubit_inds |= layout.qubit_inds  # updates dict
        return cls(setup=setup, qubit_inds=qubit_inds)

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # Split the 'for' loop in two so that the stim diagram looks better
        if self.uniform:
            prob = self.param("idle_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            prob = self.param("idle_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("idle_error_prob", qubit)
                circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

            for qubit, ind in zip(qubits, inds):
                prob = self.param("idle_error_prob", qubit)
                circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        return circ


class IncomingDepolNoiseModel(NoiselessModel):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        return Model.__init__(self, setup=setup, qubit_inds=qubit_inds)

    @classmethod
    def from_layouts(
        cls: type[IncomingDepolNoiseModel], setup: Setup, *layouts: Layout
    ) -> "IncomingDepolNoiseModel":
        """Creates a ``Model`` object using the information from the layouts."""
        qubit_inds = {}
        for layout in layouts:
            qubit_inds |= layout.qubit_inds  # updates dict
        return cls(setup=setup, qubit_inds=qubit_inds)

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        if self.uniform:
            prob = self.param("idle_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob = self.param("idle_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))

        return circ


class PhenomenologicalNoiseModel(IncomingNoiseModel):
    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MZ", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MZ", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MX", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MX", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MY", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MY", inds))
        else:
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


class PhenomenologicalDepolNoiseModel(IncomingDepolNoiseModel):
    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MZ", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MZ", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MX", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MX", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MY", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MY", inds))
        else:
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


class MeasurementNoiseModel(NoiselessModel):
    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        return Model.__init__(self, setup=setup, qubit_inds=qubit_inds)

    @classmethod
    def from_layouts(
        cls: type[MeasurementNoiseModel], setup: Setup, *layouts: Layout
    ) -> "MeasurementNoiseModel":
        """Creates a ``Model`` object using the information from the layouts."""
        qubit_inds = {}
        for layout in layouts:
            qubit_inds |= layout.qubit_inds  # updates dict
        return cls(setup=setup, qubit_inds=qubit_inds)

    def measure(self, qubits: Iterable[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # separates X_ERROR and MZ for clearer stim diagrams
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MZ", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MZ", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MX", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MX", inds))
        else:
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
        if self.uniform:
            prob = self.param("meas_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
            for qubit in qubits:
                self.add_meas(qubit)
            if self.param("assign_error_flag"):
                prob = self.param("assign_error_prob")
                circ.append(CircuitInstruction("MY", inds, [prob]))
            else:
                circ.append(CircuitInstruction("MY", inds))
        else:
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

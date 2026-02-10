from __future__ import annotations

from collections.abc import Collection, Sequence

from stim import Circuit, CircuitInstruction
from typing_extensions import override

from ..setups import (
    NLR,
    SD6,
    SI1000,
    BiasedCircuitNoiseSetup,
    CircuitNoiseSetup,
    ExtendedSI1000,
    IncomingNoiseSetup,
    IncResMeasNoiseSetup,
    MeasurementNoiseSetup,
    PhenomenologicalNoiseSetup,
    Setup,
    UniformDepolarizing,
)
from ..setups.setup import (
    LONG_RANGE_TQ_GATES,
    SQ_GATES,
    SQ_MEASUREMENTS,
    SQ_RESETS,
    TQ_GATES,
)
from .model import Model
from .util import (
    biased_prefactors,
    idle_error_probs,
    long_range_tq_gate_depol2_generator,
    sq_gate_biased_generator,
    sq_gate_depol1_generator,
    sq_gate_noiseless_generator,
    sq_meas_assign_depol1_generator,
    sq_meas_depol1_assign_generator,
    sq_meas_noiseless_generator,
    sq_meas_x_assign_generator,
    sq_reset_noiseless_generator,
    sq_reset_x_generator,
    tq_gate_biased_generator,
    tq_gate_depol1_generator,
    tq_gate_depol2_generator,
    tq_gate_noiseless_generator,
)

NOT_MEAS = SQ_GATES | TQ_GATES | LONG_RANGE_TQ_GATES | SQ_RESETS
ALL_TQ_GATES = TQ_GATES | LONG_RANGE_TQ_GATES
ALL_OPS = SQ_GATES | ALL_TQ_GATES | SQ_MEASUREMENTS | SQ_RESETS


class CircuitNoiseModel(Model):
    DEFAULT_SETUP: Setup | None = CircuitNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)
        if not callable(attr):
            return attr

        if name in SQ_GATES:
            return sq_gate_depol1_generator(self, name)
        elif name in ALL_TQ_GATES:
            return tq_gate_depol2_generator(self, name)
        elif name in SQ_MEASUREMENTS:
            return sq_meas_x_assign_generator(self, name)
        elif name in SQ_RESETS:
            return sq_reset_x_generator(self, name)
        return attr

    @override
    def idle_noise(
        self, qubits: Collection[str], param_name: str = "idle_error_prob"
    ) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        if self.uniform:
            prob: float = self.param(param_name)
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(param_name, qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))
        return circ

    @override
    def incoming_noise(self, qubits: Collection[str]) -> Circuit:
        return Circuit()


class MovableQubitsCircuitNoiseModel(CircuitNoiseModel):
    DEFAULT_SETUP: Setup | None = CircuitNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if name == "swap" and callable(attr):
            return tq_gate_depol1_generator(self, name)

        return attr


class SD6NoiseModel(CircuitNoiseModel):
    """
    The SD6 noise model is defined in the following paper:

    Gidney, C., Newman, M., & McEwen, M. (2022).
    Benchmarking the planar honeycomb code. Quantum, 6, 813.
    https://doi.org/10.22331/q-2022-09-21-813

    see Table 2 and Table 3 for a description of the noise models.

    To correctly use the SD6 noise model (i.e. with the correct error rate relations),
    one needs to use the `surface_sim.setups.SD6` setup.

    The only physical operations available in this noise model are:
    - CX
    - any single-qubit Clifford
    - initialization in the Z basis
    - measurement in the Z basis
    - idling
    """

    _supported_operations: list[str] = list(SQ_GATES) + [
        "cnot",
        "cx",
        "measure",
        "measure_z",
        "reset",
        "reset_z",
    ]

    DEFAULT_SETUP: Setup | None = SD6()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if callable(attr) and (name in ALL_OPS):
            if name not in self._supported_operations:
                raise ValueError(
                    f"Operation {name} is not available in the SD6 noise model."
                )

        return attr


class UniformDepolarizingNoiseModel(CircuitNoiseModel):
    """
    The UniformDepolarizing noise model is defined in the following paper:

    McEwen, M., Bacon, D., & Gidney, C. (2023).
    Relaxing hardware requirements for surface code circuits using time-dynamics. Quantum, 7, 1172.
    https://doi.org/10.22331/q-2023-11-07-1172

    see Table D.1 and Table D.2 for a description of the noise models.

    To correctly use the UniformDepolarizing noise model (i.e. with the correct error rate relations),
    one needs to use the `surface_sim.setups.UniformDepolarizing` setup.

    The only physical operations available in this noise model are:
    - CX
    - CXSWAP
    - any single-qubit Clifford
    - initialization in the Z and X basis
    - measurement in the Z and X basis
    - 2-qubit Pauli measuremnts
    - idling
    """

    _supported_operations: list[str] = list(SQ_GATES) + [
        "cnot",
        "cx",
        "cxswap",
        "measure",
        "measure_z",
        "measure_x",
        "reset",
        "reset_z",
        "reset_x",
    ]

    DEFAULT_SETUP: Setup | None = UniformDepolarizing()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if callable(attr) and (name in ALL_OPS):
            if name not in self._supported_operations:
                raise ValueError(
                    f"Operation {name} is not available in the SD6 noise model."
                )

            if name in SQ_MEASUREMENTS:
                return sq_meas_assign_depol1_generator(self, name)

        return attr


class SI1000NoiseModel(CircuitNoiseModel):
    """
    The SI1000 noise model is defined in the following paper:

    Gidney, C., Newman, M., & McEwen, M. (2022).
    Benchmarking the planar honeycomb code. Quantum, 6, 813.
    https://doi.org/10.22331/q-2022-09-21-813

    see Table 2 and Table 3 for a description of the noise models.

    To correctly use the SI1000 noise model (i.e. with the correct error rate relations),
    one needs to use the `surface_sim.setups.SI1000` setup.

    The only physical operations available in this noise model are:
    - CZ
    - any single-qubit Clifford
    - initialization in the Z basis
    - measurement in the Z basis
    - idling
    """

    _supported_operations: list[str] = list(SQ_GATES) + [
        "cphase",
        "cz",
        "measure",
        "measure_z",
        "reset",
        "reset_z",
    ]

    DEFAULT_SETUP: Setup | None = SI1000()

    @override
    def new_circuit(self) -> None:
        self._meas_or_reset_qubits: list[str] = []
        self._meas_reset_ops: set[str] = set(list(SQ_MEASUREMENTS) + list(SQ_RESETS))
        super().new_circuit()
        return

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if callable(attr) and (name in ALL_OPS):
            if name not in self._supported_operations:
                raise ValueError(
                    f"Operation {name} is not available in the {self.__class__.__name__} noise model."
                )

            if name in self._meas_reset_ops:

                def wrapper(qubits: Collection[str], *args, **kargs):
                    self._meas_or_reset_qubits += list(qubits)
                    return attr(qubits, *args, **kargs)

                return wrapper

        return attr

    @override
    def flush_noise(self) -> Circuit:
        circ = Circuit()
        if self._meas_or_reset_qubits:
            idle_qubits = set(self._qubit_inds).difference(self._meas_or_reset_qubits)
            circ += self.idle_noise(idle_qubits, "extra_idle_meas_or_reset_error_prob")
        self._meas_or_reset_qubits = []
        return circ


class ExtendedSI1000NoiseModel(CircuitNoiseModel):
    """
    The (extended) SI1000 noise model is defined in the following paper:

    McEwen, M., Bacon, D., & Gidney, C. (2023).
    Relaxing hardware requirements for surface code circuits using time-dynamics. Quantum, 7, 1172.
    https://doi.org/10.22331/q-2023-11-07-1172

    see Table D.1 and Table D.2 for a description of the noise models.

    To correctly use the ExtendedSI1000 noise model (i.e. with the correct error rate relations),
    one needs to use the `surface_sim.setups.ExtendedSI1000` setup.

    The only physical operations available in this noise model are:
    - CZ
    - iSWAP
    - any single-qubit Clifford
    - initialization in the Z basis
    - measurement in the Z basis
    - ZZ measurement
    - idling
    """

    _supported_operations: list[str] = list(SQ_GATES) + [
        "cphase",
        "iswap",
        "cz",
        "measure",
        "measure_z",
        "reset",
        "reset_z",
    ]

    DEFAULT_SETUP: Setup | None = ExtendedSI1000()

    @override
    def new_circuit(self) -> None:
        self._meas_or_reset_qubits: list[str] = []
        self._meas_reset_ops: set[str] = set(list(SQ_MEASUREMENTS) + list(SQ_RESETS))
        super().new_circuit()
        return

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if callable(attr) and (name in ALL_OPS):
            if name not in self._supported_operations:
                raise ValueError(
                    f"Operation {name} is not available in the {self.__class__.__name__} noise model."
                )

            if name in SQ_MEASUREMENTS:
                attr = sq_meas_assign_depol1_generator(self, name)

            if name in self._meas_reset_ops:

                def wrapper(qubits: Collection[str], *args, **kargs):
                    self._meas_or_reset_qubits += list(qubits)
                    return attr(qubits, *args, **kargs)

                return wrapper

        return attr

    @override
    def flush_noise(self) -> Circuit:
        circ = Circuit()
        if self._meas_or_reset_qubits:
            idle_qubits = set(self._qubit_inds).difference(self._meas_or_reset_qubits)
            circ += self.idle_noise(idle_qubits, "extra_idle_meas_or_reset_error_prob")
        self._meas_or_reset_qubits = []
        return circ


class NLRNoiseModel(SI1000NoiseModel):
    """
    The NLR noise model is defined in the following paper:

    Beni, L. A., Higgott, O., & Shutty, N. (2025).
    Tesseract: A search-based decoder for quantum error correction.
    arXiv preprint arXiv:2503.10988.

    which corresponds to a ``SI1000`` noise model but with higher noise
    strength for the two-qubit depolarizing channels after the long-range CZ gates.

    See the documentation for the ``SI1000NoiseModel`` for more information.
    """

    DEFAULT_SETUP: Setup | None = NLR()

    def __init__(
        self,
        qubit_inds: dict[str, int],
        qubit_coords: dict[str, Sequence[float | int]],
        setup: Setup | None = None,
    ) -> None:
        self._qubit_coords: dict[str, Sequence[float | int]] = qubit_coords
        super().__init__(qubit_inds=qubit_inds, setup=setup)
        return

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if callable(attr) and (name in ("cz", "cphase")):
            return long_range_tq_gate_depol2_generator(self, name)

        return attr


class BiasedCircuitNoiseModel(Model):
    DEFAULT_SETUP: Setup | None = BiasedCircuitNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if not callable(attr):
            return attr

        if name in SQ_GATES:
            return sq_gate_biased_generator(self, name)
        elif name in TQ_GATES:
            return tq_gate_biased_generator(self, name)
        elif name in SQ_MEASUREMENTS:
            return sq_meas_x_assign_generator(self, name)
        elif name in SQ_RESETS:
            return sq_reset_x_generator(self, name)
        return attr

    @override
    def idle(self, qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        circ.append(CircuitInstruction("I", inds))
        circ += self.idle_noise(qubits)

        return circ

    @override
    def idle_noise(
        self, qubits: Collection[str], param_name: str = "idle_error_prob"
    ) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        if self.uniform:
            prob: float = self.param(param_name)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli"),
                biased_factor=self.param("biased_factor"),
                num_qubits=1,
            )
            prob = prob * prefactors
            circ.append(CircuitInstruction("PAULI_CHANNEL_1", inds, prob))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param(param_name, qubit)
                prefactors = biased_prefactors(
                    biased_pauli=self.param("biased_pauli", qubit),
                    biased_factor=self.param("biased_factor", qubit),
                    num_qubits=1,
                )
                prob = prob * prefactors
                circ.append(CircuitInstruction("PAULI_CHANNEL_1", [ind], prob))
        return circ

    @override
    def incoming_noise(self, qubits: Collection[str]) -> Circuit:
        return Circuit()


class T1T2NoiseModel(Model):
    """A coherence-limited noise model using T1 and T2.
    The noise is added when perfoming gates and when calling
    ``T1T2NoiseModel.tick``.
    """

    @override
    def new_circuit(self) -> None:
        self._durations: dict[str, float] = {q: 0.0 for q in self._qubit_inds}
        super().new_circuit()
        return

    def _generic_gate(self, name: str, qubits: Sequence[str]) -> Circuit:
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

    def _generic_measurement(self, name: str, qubits: Collection[str]) -> Circuit:
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

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if not callable(attr):
            return attr

        if name != "idle" and (name in NOT_MEAS):

            def sq_gate(qubits: Collection[str]) -> Circuit:
                for qubit in qubits:
                    self._durations[qubit] += self.gate_duration(NOT_MEAS[name])
                return self._generic_gate(NOT_MEAS[name], qubits)

            return sq_gate

        elif name in SQ_MEASUREMENTS:

            def sq_meas(qubits: Collection[str]) -> Circuit:
                for qubit in qubits:
                    self._durations[qubit] += self.gate_duration(SQ_MEASUREMENTS[name])
                return self._generic_measurement(SQ_MEASUREMENTS[name], qubits)

            return sq_meas

        return attr

    @override
    def idle(self, qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()
        circ.append(CircuitInstruction("I", inds))
        return circ

    @override
    def idle_noise(self, qubits: Collection[str]) -> Circuit:
        return Circuit()

    @override
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

    def idle_duration(self, qubits: Collection[str], duration: float) -> Circuit:
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
            relax_time: float = self.param("T1")
            deph_time: float = self.param("T2")
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
            relax_time: float = self.param("T1", qubit)
            deph_time: float = self.param("T2", qubit)
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

    @override
    def incoming_noise(self, qubits: Collection[str]) -> Circuit:
        return Circuit()


class NoiselessModel(Model):
    DEFAULT_SETUP: Setup | None = Setup(dict(setup=[{}]))

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if not callable(attr):
            return attr

        if name in SQ_GATES:
            return sq_gate_noiseless_generator(self, name)
        elif name in TQ_GATES:
            return tq_gate_noiseless_generator(self, name)
        elif name in SQ_MEASUREMENTS:
            return sq_meas_noiseless_generator(self, name)
        elif name in SQ_RESETS:
            return sq_reset_noiseless_generator(self, name)
        return attr

    @override
    def idle_noise(self, qubits: Collection[str]) -> Circuit:
        return Circuit()

    @override
    def incoming_noise(self, qubits: Collection[str]) -> Circuit:
        return Circuit()


class IncomingNoiseModel(NoiselessModel):
    DEFAULT_SETUP: Setup | None = IncomingNoiseSetup()

    @override
    def incoming_noise(self, qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        # Split the 'for' loop in two so that the stim diagram looks better
        if self.uniform:
            prob: float = self.param("idle_error_prob")
            circ.append(CircuitInstruction("X_ERROR", inds, [prob]))
            prob: float = self.param("idle_error_prob")
            circ.append(CircuitInstruction("Z_ERROR", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param("idle_error_prob", qubit)
                circ.append(CircuitInstruction("X_ERROR", [ind], [prob]))

            for qubit, ind in zip(qubits, inds):
                prob: float = self.param("idle_error_prob", qubit)
                circ.append(CircuitInstruction("Z_ERROR", [ind], [prob]))

        return circ


class IncomingDepolNoiseModel(NoiselessModel):
    DEFAULT_SETUP: Setup | None = IncomingNoiseSetup()

    @override
    def incoming_noise(self, qubits: Collection[str]) -> Circuit:
        inds = self.get_inds(qubits)
        circ = Circuit()

        if self.uniform:
            prob: float = self.param("idle_error_prob")
            circ.append(CircuitInstruction("DEPOLARIZE1", inds, [prob]))
        else:
            for qubit, ind in zip(qubits, inds):
                prob: float = self.param("idle_error_prob", qubit)
                circ.append(CircuitInstruction("DEPOLARIZE1", [ind], [prob]))

        return circ


class PhenomenologicalNoiseModel(IncomingNoiseModel):
    DEFAULT_SETUP: Setup | None = PhenomenologicalNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if (name in SQ_MEASUREMENTS) and callable(attr):
            return sq_meas_x_assign_generator(self, name)

        return attr


class PhenomenologicalDepolNoiseModel(IncomingDepolNoiseModel):
    DEFAULT_SETUP: Setup | None = PhenomenologicalNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if (name in SQ_MEASUREMENTS) and callable(attr):
            return sq_meas_depol1_assign_generator(self, name)

        return attr


class IncResMeasNoiseModel(PhenomenologicalNoiseModel):
    DEFAULT_SETUP: Setup | None = IncResMeasNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if (name in SQ_RESETS) and callable(attr):
            return sq_reset_x_generator(self, name)

        return attr


class MeasurementNoiseModel(NoiselessModel):
    DEFAULT_SETUP: Setup | None = MeasurementNoiseSetup()

    @override
    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if (name in SQ_MEASUREMENTS) and callable(attr):
            return sq_meas_x_assign_generator(self, name)

        return attr

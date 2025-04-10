from collections.abc import Sequence, Iterable

from stim import CircuitInstruction, target_rec, GateTarget, Circuit

from ..setup import Setup


class Model(object):
    """Noise model class for generating the ``stim.Circuit``s for each
    of the physical operations including noise channels.

    **IMPORTANT**

    It assumes that layers of operations are separated by ``Model.tick()``,
    and that all qubits participiate in an operation (for idling, use e.g.
    ``Model.idle``) if they are active (i.e. not measured).

    When designing new noise model classes, the output should be a ``stim.Circuit``
    that must include the operation of the corresponding class method
    (e.g. ``"X"`` for ``Model.x_gate``) and (optionally) noise channels.
    It should not include anything else.
    """

    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        self._setup = setup
        self._qubit_inds = qubit_inds
        self._meas_order = {q: [] for q in qubit_inds}
        self._num_meas = 0
        return

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def qubits(self) -> list[str]:
        return list(self._qubit_inds.keys())

    @property
    def uniform(self) -> bool:
        return self._setup.uniform

    def gate_duration(self, name: str) -> float:
        return self._setup.gate_duration(name)

    def get_inds(self, qubits: Iterable[str]) -> list[object]:
        # The proper annotation for this function should be "-> list[int]"
        # but stim gets confused and only accepts list[object] making the
        # LSP unusable with all the errors.
        return [self._qubit_inds[q] for q in qubits]

    def param(self, *args, **kargs):
        return self._setup.param(*args, **kargs)

    # easier detector definition
    def add_meas(self, qubit: str) -> None:
        """Adds a measurement record for the specified qubit.
        This information is used in the ``meas_target`` function.
        """
        if qubit not in self._qubit_inds:
            raise ValueError(f"{qubit} is not in the specified qubit_inds.")

        self._meas_order[qubit].append(self._num_meas)
        self._num_meas += 1
        return

    def meas_target(self, qubit: str, rel_meas_ind: int) -> GateTarget:
        """Returns the global measurement index for ``stim.target_rec`` for the
        specified qubit and its relative measurement index
        (for the given qubit).

        Instead of working with global measurement indexing (as ``stim`` does),
        this function allows to work with local measurement indexing for
        each qubit (see ``Notes`` for an example).

        Parameters
        ----------
        qubit
            Label of the qubit.
        rel_meas_ind
            Relative measurement index for the given qubit.

        Returns
        -------
        GateTarget
            Target measurement index (``stim.target_rec``) for building the
            detectors and observables.

        Notes
        -----
        To access the first measurement in the following example

        .. codeblock::

            M 0
            M 1
            M 0

        one would use ``-3`` in ``stim``'s indexing. However, in the "local
        measurement indexing for each qubit", it would correspond to (q0, -2).
        This makes it easier for building the detectors as they correspond to
        the XOR of syndrome outcomes (ancilla outcomes) between different QEC
        rounds (or some linear combination of syndrome outcomes).
        """
        num_meas_qubit = len(self._meas_order[qubit])
        if (rel_meas_ind > num_meas_qubit) or (rel_meas_ind < -num_meas_qubit):
            raise ValueError(
                f"{qubit} has only {num_meas_qubit} measurements, but {rel_meas_ind} was accessed."
            )

        abs_meas_ind = self._meas_order[qubit][rel_meas_ind]
        return target_rec(abs_meas_ind - self._num_meas)

    def new_circuit(self) -> None:
        """Empties the variables used for ``meas_target``. This must be called
        when creating a new circuit."""
        self.__init__(setup=self._setup, qubit_inds=self._qubit_inds)
        return

    # annotation operations
    def tick(self) -> Circuit:
        return Circuit("TICK")

    def qubit_coords(self, coords: dict[str, list]) -> Circuit:
        if set(coords) > set(self._qubit_inds):
            raise ValueError(
                "'coords' have qubits not defined in the model:\n"
                f"coords={list(coords.keys())}\nmodel={list(self._qubit_inds.keys())}."
            )

        circ = Circuit()

        # sort the qubit coordinate definitions by index so that it is reproducible
        ind_coord_pairs = [(self._qubit_inds[label], c) for label, c in coords.items()]
        ind_coord_pairs.sort(key=lambda x: x[0])
        for q_ind, q_coords in ind_coord_pairs:
            circ.append(CircuitInstruction("QUBIT_COORDS", [q_ind], q_coords))

        return circ

    # gate/measurement/reset operations
    def x_gate(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def z_gate(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def hadamard(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def s_gate(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def s_dag_gate(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def cnot(self, qubits: Sequence[str]) -> Circuit:
        raise NotImplementedError

    def cphase(self, qubits: Sequence[str]) -> Circuit:
        raise NotImplementedError

    def swap(self, qubits: Sequence[str]) -> Circuit:
        raise NotImplementedError

    def measure(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def measure_x(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def measure_y(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def measure_z(self, qubits: Iterable[str]) -> Circuit:
        return self.measure(qubits)

    def reset(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def reset_x(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def reset_y(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def reset_z(self, qubits: Iterable[str]) -> Circuit:
        return self.reset(qubits)

    def idle(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def idle_meas(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def idle_reset(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def idle_noise(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

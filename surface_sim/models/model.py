from typing import Any, Dict, List, Iterator, Iterable

from stim import CircuitInstruction, target_rec, GateTarget

from ..setup import Setup


class Model(object):
    def __init__(self, setup: Setup, qubit_inds: Dict[str, int]) -> None:
        self._setup = setup
        self._qubit_inds = qubit_inds
        self._meas_order = {q: [] for q in qubit_inds}
        self._num_meas = 0
        return

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def qubits(self) -> List[str]:
        return list(self._qubit_inds.keys())

    def gate_duration(self, name: str) -> float:
        return self._setup.gate_duration(name)

    def get_inds(self, qubits: Iterable[str]) -> List[object]:
        # The proper annotation for this function should be "-> List[int]"
        # but stim gets confused and only accepts List[object] making the
        # LSP unusable with all the errors.
        return [self._qubit_inds[q] for q in qubits]

    def param(self, *qubits: str) -> Any:
        return self._setup.param(*qubits)

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
        self._meas_order = {q: [] for q in self._qubit_inds}
        self._num_meas = 0
        return

    # annotation operations
    def tick(self, *args, **kargs):
        yield CircuitInstruction("TICK", targets=[])

    def qubit_coords(self, coords: Dict[str, list]):
        if set(coords) > set(self._qubit_inds):
            raise ValueError(
                "'coords' have qubits not defined in the model:\n"
                f"coords={list(coords.keys())}\nmodel={list(self._qubit_inds.keys())}."
            )

        for q_label, q_coords in coords.items():
            q_ind = self._qubit_inds[q_label]
            yield CircuitInstruction("QUBIT_COORDS", [q_ind], q_coords)

    # gate/measurement/reset operations
    def x_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def z_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def hadamard(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def s_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def s_dag_gate(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def cphase(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def measure(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def reset(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

    def idle(self, qubits: Iterable[str]) -> Iterator[CircuitInstruction]:
        raise NotImplementedError

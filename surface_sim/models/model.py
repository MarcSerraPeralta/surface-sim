from __future__ import annotations
from collections.abc import Sequence, Iterable

from copy import deepcopy

from stim import CircuitInstruction, target_rec, GateTarget, Circuit

from ..setup import Setup
from ..layouts import Layout


class Model:
    """Noise model class for generating the ``stim.Circuit``s for each
    of the physical operations including noise channels.

    **IMPORTANT**

    The noise models assume that operation layers are separated by ``Model.tick()``,
    and that all qubits participiate in an operation in the opertion layers.
    Note that ``Model.idling`` is considered an operation, i.e. ``"I"``.

    When designing new noise model classes,

    1. the method output should be a ``stim.Circuit`` that must include the operation
    of the corresponding method (e.g. ``"X"`` for ``Model.x_gate``) and
    (optionally) noise channels. It should not include anything else.

    2. ``Model.tick``s do not contain any noise except from the one called by
    ``Model.flush_noise``. ``Model.flush_noise`` adds all the "still-not-added"
    noise from the previous operation layer (this is useful in e.g. ``DecoherenceNoiseModel``).
    Note that if ``Model.tick`` are followed one after the other, ``Model.flush_noise``
    is only called for the first one. This is done so that there are no issues when
    merging operation layers and because TICKs are just annotations, not noise.
    If noise wants to be present between TICKs, then idling gates must be added.

    For more information, read the comments in issue #232.
    """

    operations = [
        "tick",
        "qubit_coords",
        "x_gate",
        "z_gate",
        "hadamard",
        "s_gate",
        "s_dag_gate",
        "cnot",
        "cphase",
        "swap",
        "measure",
        "measure_x",
        "measure_y",
        "measure_z",
        "reset",
        "reset_x",
        "reset_y",
        "reset_z",
        "idle",
        "idle_meas",
        "idle_reset",
    ]

    def __init__(self, setup: Setup, qubit_inds: dict[str, int]) -> None:
        self._setup = setup
        self._qubit_inds = qubit_inds
        self._meas_order = {q: [] for q in qubit_inds}
        self._num_meas = 0
        self._last_op = ""
        self._new_op = ""
        return

    @classmethod
    def from_layouts(cls: type[Model], setup: Setup, *layouts: Layout) -> "Model":
        """Creates a ``Model`` object using the information from the layouts."""
        qubit_inds = {}
        for layout in layouts:
            qubit_inds |= layout.qubit_inds  # updates dict
        return cls(setup=setup, qubit_inds=qubit_inds)

    def __getattribute__(self, name):
        """
        Stores the name of the last operation called in this class.
        The operations include: annotations, gates, measurements and resets.
        """
        attr = object.__getattribute__(self, name)

        if callable(attr) and (name in self.operations):

            def wrapper(*args, **kwargs):
                # this function is before running the called method.
                # if I only store the last operation it will be overwritten
                # by the new called method, thus I need to store the last and
                # new operations.
                self._last_op = deepcopy(self._new_op)
                self._new_op = name
                return attr(*args, **kwargs)

            return wrapper

        return attr

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
        if self._last_op != "tick":
            missing_noise = self.flush_noise()
            return missing_noise + Circuit("TICK")
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

    # noise methods
    def flush_noise(self) -> Circuit:
        return Circuit()

    def idle_noise(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

    def incoming_noise(self, qubits: Iterable[str]) -> Circuit:
        raise NotImplementedError

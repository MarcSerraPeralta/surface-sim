"""
Decorators for functions that
1. take ``model: Model`` and ``layout: Layout`` as inputs (nothing else)
2. return a generator the iterates over stim.Circuit(s)
"""

from collections.abc import Callable, Generator
from copy import deepcopy

import stim

from ..layouts import Layout
from ..models import Model

LogOpFunction = (
    Callable[[Model, Layout], Generator[stim.Circuit]]
    | Callable[[Model, Layout, Layout], Generator[stim.Circuit]]
)


class LogOpCallable:
    def __init__(self, func: LogOpFunction):
        self.func: LogOpFunction = func
        self.log_op_type: list[str] = []
        self.rot_basis: bool | None = None
        self.num_qubits: int | None = None
        self.noiseless: bool = False
        self.name: str = func.__name__
        self.anc_reset: bool | None = None
        self.log_noise_prob: float | None = None
        self.pauli_observable_ind: int | None = None
        return

    def __call__(self, *args, **kargs) -> Generator[stim.Circuit]:
        if self.anc_reset is not None:
            kargs |= {"anc_reset": self.anc_reset}
        if self.log_noise_prob is not None:
            kargs |= {"prob": self.log_noise_prob}
        if self.pauli_observable_ind is not None:
            kargs |= {"observable_ind": self.pauli_observable_ind}
        if not self.noiseless:
            yield from self.func(*args, **kargs)
        else:
            for c in self.func(*args, **kargs):
                yield c.without_noise()

    @property
    def __name__(self) -> str:
        return self.name

    def copy(self) -> "LogOpCallable":
        new_copy = LogOpCallable(deepcopy(self.func))
        new_copy.log_op_type = deepcopy(self.log_op_type)
        new_copy.rot_basis = deepcopy(self.rot_basis)
        new_copy.num_qubits = deepcopy(self.num_qubits)
        new_copy.noiseless = deepcopy(self.noiseless)
        new_copy.name = deepcopy(self.name)
        new_copy.anc_reset = deepcopy(self.anc_reset)
        new_copy.log_noise_prob = deepcopy(self.log_noise_prob)
        new_copy.pauli_observable_ind = deepcopy(self.pauli_observable_ind)
        return new_copy

    def __str__(self) -> str:
        string = self.name
        if self.pauli_observable_ind is not None:
            string += f"({self.pauli_observable_ind})"
        if self.rot_basis is not None:
            string += " bX" if self.rot_basis else " bZ"
        if self.anc_reset is not None:
            string += f" anc_reset={self.anc_reset}"
        if self.num_qubits is not None:
            string += f" n={self.num_qubits}"
        if self.log_noise_prob is not None:
            string += f" log-noise-prob={self.log_noise_prob}"
        if self.noiseless is True:
            string += " noiseless"
        return f"'{string}'"

    def __repr__(self) -> str:
        return str(self)


LogicalOperation = tuple[LogOpCallable, Layout] | tuple[LogOpCallable, Layout, Layout]


def qec_circuit(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["qec_round"]
    return new_func


def to_mid_cycle_circuit(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["to_mid_cycle_circuit"]
    return new_func


def to_end_cycle_circuit(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["to_end_cycle_circuit"]
    return new_func


def sq_gate(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["sq_unitary_gate"]
    new_func.num_qubits = 1
    return new_func


def tq_gate(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["tq_unitary_gate"]
    new_func.rot_basis = None
    new_func.num_qubits = 2
    return new_func


def qubit_init_z(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["qubit_init"]
    new_func.rot_basis = False
    return new_func


def qubit_init_x(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["qubit_init"]
    new_func.rot_basis = True
    return new_func


def qubit_encoding(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["qubit_encoding"]
    return new_func


def logical_measurement_z(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["measurement"]
    new_func.rot_basis = False
    return new_func


def logical_measurement_x(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["measurement"]
    new_func.rot_basis = True
    return new_func


def logical_noise(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["logical_noise"]
    return new_func


def pauli_observable_x(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["pauli_observable"]
    return new_func


def pauli_observable_y(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["pauli_observable"]
    return new_func


def pauli_observable_z(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.log_op_type += ["pauli_observable"]
    return new_func


def noiseless(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    """Decorator for removing all noise channels from a ``LogOpCallable``"""
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    new_func = func.copy()
    new_func.noiseless = True
    return new_func


def copy_from(
    other_func: LogOpCallable,
) -> Callable[[LogOpCallable | LogOpFunction], LogOpCallable]:
    """Decorator for copying all the logical operation attributes from a function."""

    def decorator(
        func: LogOpCallable | LogOpFunction, other_func=other_func.copy()
    ) -> LogOpCallable:
        other_func.func = func
        return other_func

    return decorator

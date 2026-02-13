"""
Decorators for functions that
1. take ``model: Model`` and ``layout: Layout`` as inputs (nothing else)
2. return a generator the iterates over stim.Circuit(s)
"""

from collections.abc import Callable, Generator

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
        return

    def __call__(self, *args, **kargs) -> Generator[stim.Circuit]:
        return self.func(*args, **kargs)

    @property
    def __name__(self) -> str:
        return self.name


LogicalOperation = tuple[LogOpCallable, Layout] | tuple[LogOpCallable, Layout, Layout]


def qec_circuit(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["qec_round"]
    return func


def to_mid_cycle_circuit(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["to_mid_cycle_circuit"]
    return func


def to_end_cycle_circuit(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["to_end_cycle_circuit"]
    return func


def sq_gate(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["sq_unitary_gate"]
    func.num_qubits = 1
    return func


def tq_gate(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["tq_unitary_gate"]
    func.rot_basis = None
    func.num_qubits = 2
    return func


def qubit_init_z(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["qubit_init"]
    func.rot_basis = False
    return func


def qubit_init_x(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["qubit_init"]
    func.rot_basis = True
    return func


def qubit_encoding(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["qubit_encoding"]
    return func


def logical_measurement_z(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["measurement"]
    func.rot_basis = False
    return func


def logical_measurement_x(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["measurement"]
    func.rot_basis = True
    return func


def logical_noise(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)
    func.log_op_type += ["logical_noise"]
    return func


def noiseless(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
    """Decorator for removing all noise channels from a ``LogOpCallable``"""
    if not isinstance(func, LogOpCallable):
        func = LogOpCallable(func)

    def noiseless_func(
        model: Model, layout: Layout, **kargs
    ) -> Generator[stim.Circuit]:
        for c in func(model, layout, **kargs):
            yield c.without_noise()

    noiseless_op = LogOpCallable(noiseless_func)

    noiseless_op.name = func.__name__
    noiseless_op.log_op_type = func.log_op_type
    noiseless_op.rot_basis = func.rot_basis
    noiseless_op.num_qubits = func.num_qubits
    noiseless_op.noiseless = True

    return noiseless_op


def copy_from(
    other_func: LogOpCallable,
) -> Callable[[LogOpCallable | LogOpFunction], LogOpCallable]:
    """Decorator for copying all the logical operation attributes from a function."""

    def decorator(func: LogOpCallable | LogOpFunction) -> LogOpCallable:
        if not isinstance(func, LogOpCallable):
            func = LogOpCallable(func)

        func.log_op_type = other_func.log_op_type
        func.rot_basis = other_func.rot_basis
        func.num_qubits = other_func.num_qubits
        func.noiseless = other_func.noiseless
        func.name = other_func.__name__

        return func

    return decorator

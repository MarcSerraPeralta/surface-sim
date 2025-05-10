"""
Decorators for functions that
1. take ``model: Model`` and ``layout: Layout`` as inputs (nothing else)
2. return a generator the iterates over stim.Circuit(s)
"""

from collections.abc import Generator
from typing import Protocol, runtime_checkable

import stim

from ..models import Model
from ..layouts import Layout


@runtime_checkable
class LogOpCallable(Protocol):
    __name__: str
    log_op_type: str
    rot_basis: bool | None
    num_qubits: int | None
    noiseless: bool

    def __call__(
        self, model: Model, layout: Layout, **kargs
    ) -> Generator[stim.Circuit]: ...


LogicalOperation = tuple[LogOpCallable, Layout] | tuple[LogOpCallable, Layout, Layout]


def qec_circuit(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"qec_round"`` to a function.
    """
    func.log_op_type = "qec_round"
    func.rot_basis = None
    func.num_qubits = None
    func.noiseless = False
    return func


def sq_gate(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"sq_unitary_gate"`` to a function.
    """
    func.log_op_type = "sq_unitary_gate"
    func.rot_basis = None
    func.num_qubits = 1
    func.noiseless = False
    return func


def tq_gate(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"tq_unitary_gate"`` to a function.
    """
    func.log_op_type = "tq_unitary_gate"
    func.rot_basis = None
    func.num_qubits = 2
    func.noiseless = False
    return func


def qubit_init_z(func):
    """
    Decorator for adding the attribute ``"log_op_type", "rot_basis"`` and setting
    them to ``"qubit_init", False`` (respectively) to a function.
    """
    func.log_op_type = "qubit_init"
    func.rot_basis = False
    func.num_qubits = None
    func.noiseless = False
    return func


def qubit_init_x(func):
    """
    Decorator for adding the attribute ``"log_op_type", "rot_basis"`` and setting
    them to ``"qubit_init", False`` (respectively) to a function.
    """
    func.log_op_type = "qubit_init"
    func.rot_basis = True
    func.num_qubits = None
    func.noiseless = False
    return func


def logical_measurement_z(func):
    """
    Decorator for adding the attributes ``"log_op_type", "rot_basis"`` and setting
    them to ``"measurement", False`` (respectively) to a function.
    """
    func.log_op_type = "measurement"
    func.rot_basis = False
    func.num_qubits = None
    func.noiseless = False
    return func


def logical_measurement_x(func):
    """
    Decorator for adding the attributes ``"log_op_type", "rot_basis"`` and setting
    them to ``"measurement", True`` (respectively) to a function.
    """
    func.log_op_type = "measurement"
    func.rot_basis = True
    func.num_qubits = None
    func.noiseless = False
    return func


def noiseless(func: LogOpCallable) -> LogOpCallable:
    """Decorator for removing all noise channels from a ``LogOpCallable``"""

    def noiseless_func(
        model: Model, layout: Layout, **kargs
    ) -> Generator[stim.Circuit]:
        for c in func(model, layout, **kargs):
            yield c.without_noise()

    noiseless_func.__name__ = func.__name__
    noiseless_func.log_op_type = func.log_op_type
    noiseless_func.rot_basis = func.rot_basis
    noiseless_func.num_qubits = func.num_qubits
    noiseless_func.noiseless = True

    return noiseless_func

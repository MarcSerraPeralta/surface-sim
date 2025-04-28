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

    def __call__(
        self, model: Model, layout: Layout, **kargs
    ) -> Generator[stim.Circuit]: ...


LogicalOperation = tuple[LogOpCallable, Layout] | tuple[LogOpCallable, Layout, Layout]


def qec_circuit(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"qec_cycle"`` to a function.
    """
    func.log_op_type = "qec_cycle"
    func.rot_basis = None
    func.num_qubits = None
    return func


def sq_gate(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"sq_unitary_gate"`` to a function.
    """
    func.log_op_type = "sq_unitary_gate"
    func.rot_basis = None
    func.num_qubits = 1
    return func


def tq_gate(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"tq_unitary_gate"`` to a function.
    """
    func.log_op_type = "tq_unitary_gate"
    func.rot_basis = None
    func.num_qubits = 2
    return func


def qubit_init_z(func):
    """
    Decorator for adding the attribute ``"log_op_type", "rot_basis"`` and setting
    them to ``"qubit_init", False`` (respectively) to a function.
    """
    func.log_op_type = "qubit_init"
    func.rot_basis = False
    func.num_qubits = None
    return func


def qubit_init_x(func):
    """
    Decorator for adding the attribute ``"log_op_type", "rot_basis"`` and setting
    them to ``"qubit_init", False`` (respectively) to a function.
    """
    func.log_op_type = "qubit_init"
    func.rot_basis = True
    func.num_qubits = None
    return func


def logical_measurement_z(func):
    """
    Decorator for adding the attributes ``"log_op_type", "rot_basis"`` and setting
    them to ``"measurement", False`` (respectively) to a function.
    """
    func.log_op_type = "measurement"
    func.rot_basis = False
    func.num_qubits = None
    return func


def logical_measurement_x(func):
    """
    Decorator for adding the attributes ``"log_op_type", "rot_basis"`` and setting
    them to ``"measurement", True`` (respectively) to a function.
    """
    func.log_op_type = "measurement"
    func.rot_basis = True
    func.num_qubits = None
    return func

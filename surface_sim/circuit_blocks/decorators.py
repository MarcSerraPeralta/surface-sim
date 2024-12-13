"""
Decorators for functions that
1. take ``model: Model`` and ``layout: Layout`` as inputs (nothing else)
2. return a generator the iterates over stim.Circuit(s)
"""


def qec_circuit(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"qec_cycle"`` to a function.
    """
    func.log_op_type = "qec_cycle"
    return func


def logical_gate(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"unitary_gate"`` to a function.
    """
    func.log_op_type = "unitary_gate"
    return func


def qubit_initialization(func):
    """
    Decorator for adding the attribute ``"log_op_type"`` and setting it to
    ``"qubit_init"`` to a function.
    """
    func.log_op_type = "qubit_init"
    return func


def logical_measurement_z(func):
    """
    Decorator for adding the attributes ``"log_op_type", "rot_basis"`` and setting
    them to ``"measurement", False`` (respectively) to a function.
    """
    func.log_op_type = "measurement"
    func.rot_basis = False
    return func


def logical_measurement_x(func):
    """
    Decorator for adding the attributes ``"log_op_type", "rot_basis"`` and setting
    them to ``"measurement", True`` (respectively) to a function.
    """
    func.log_op_type = "measurement"
    func.rot_basis = True
    return func

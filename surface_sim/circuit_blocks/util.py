from typing import Dict
import warnings
from itertools import chain

from stim import Circuit

from ..layouts import Layout
from ..models import Model
from ..detectors import Detectors


def qubit_coords(model: Model, layout: Layout) -> Circuit:
    """Returns a stim circuit that sets up the coordinates
    of the qubits.
    """
    coord_dict = {q: layout.get_coords([q])[0] for q in layout.get_qubits()}
    circuit = Circuit()

    for instruction in model.qubit_coords(coord_dict):
        circuit.append(instruction)

    return circuit


def log_meas(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical measurement
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    circuit = Circuit()

    if rot_basis:
        for instruction in model.hadamard(data_qubits):
            circuit.append(instruction)

        for instruction in model.idle(anc_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)

    circuit.append("TICK")

    # detectors and logical observables
    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    detectors_stim = detectors.build_from_data(
        model.meas_target, layout.adjacency_matrix(), meas_reset, anc_qubits=stabs
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    if log_op not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_x and log_z in your layout.",
            DeprecationWarning,
        )
        targets = [model.meas_target(qubit, -1) for qubit in data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    else:
        log_data_qubits = getattr(layout, log_op)
        targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def init_qubits(
    model: Model,
    layout: Layout,
    data_init: Dict[str, int],
    rot_basis: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical initialization
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()
    for instruction in model.reset(qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    exc_qubits = set([q for q, s in data_init.items() if s])
    if exc_qubits:
        for instruction in model.x_gate(exc_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - exc_qubits
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    if rot_basis:
        for instruction in model.hadamard(data_qubits):
            circuit.append(instruction)
        for instruction in model.idle(anc_qubits):
            circuit.append(instruction)
        circuit.append("TICK")

    return circuit


def log_x(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical X gate
    of the given model.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    if "log_x" not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_x in your layout.",
            DeprecationWarning,
        )
        log_x_qubits = data_qubits
    else:
        log_x_qubits = layout.log_x

    circuit = Circuit()

    for instruction in model.x_gate(log_x_qubits):
        circuit.append(instruction)

    idle_qubits = set(anc_qubits) + set(data_qubits) - set(log_x_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # the stabilizer generators do not change when applying a logical X gate

    return circuit


def log_z(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical Z gate
    of the given model.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    if "log_z" not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_z in your layout.",
            DeprecationWarning,
        )
        log_z_qubits = data_qubits
    else:
        log_z_qubits = layout.log_z

    circuit = Circuit()

    for instruction in model.z_gate(log_z_qubits):
        circuit.append(instruction)

    idle_qubits = set(anc_qubits) + set(data_qubits) - set(log_z_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # the stabilizer generators do not change when applying a logical Z gate

    return circuit


def log_trans_s(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """Returns the stim circuit corresponding to a transversal logical S gate
    implemented following:

    https://quantum-journal.org/papers/q-2024-04-08-1310/
    """
    qubits = set(layout.get_qubits())

    cz_pairs = set()
    qubits_s_gate = set()
    qubits_s_dag_gate = set()
    for data_qubit in layout.get_qubits(role="data"):
        trans_s = layout.param("trans_s", data_qubit)
        if trans_s is None:
            raise ValueError(
                "The layout does not have the information to run a "
                f"transversal S gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        # Using a set to avoid repetition of the cz gates.
        # Using tuple so that the object is hashable for the set.
        cz_pairs.add(tuple(sorted([data_qubit, trans_s["cz"]])))
        if trans_s["local"] == "S":
            qubits_s_gate.add(data_qubit)
        elif trans_s["local"] == "S_DAG":
            qubits_s_dag_gate.add(data_qubit)

    circuit = Circuit()

    # S, S_DAG gates
    idle_qubits = qubits - qubits_s_gate - qubits_s_dag_gate
    for instruction in model.s_gate(qubits_s_gate):
        circuit.append(instruction)
    for instruction in model.s_dag_gate(qubits_s_dag_gate):
        circuit.append(instruction)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # long-range CZ gates
    int_qubits = list(chain.from_iterable(cz_pairs))
    idle_qubits = qubits - set(int_qubits)
    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # update the stabilizer generators
    unitary_mat = layout.stab_gen_matrix("trans_s")
    detectors.update(unitary_mat)

    return circuit


def log_meas_xzzx(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical measurement
    of the given model.

    Parameters
    ----------
    rot_basis
        If True, the logical measurement is in the X basis.
        By default, the logical measurement is in the Z basis.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits

    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)

    circuit.append("TICK")

    # detectors and logical observables
    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    detectors_stim = detectors.build_from_data(
        model.meas_target, layout.adjacency_matrix(), meas_reset, anc_qubits=stabs
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    if log_op not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_x and log_z in your layout.",
            DeprecationWarning,
        )
        targets = [model.meas_target(qubit, -1) for qubit in data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    else:
        log_data_qubits = getattr(layout, log_op)
        targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def init_qubits_xzzx(
    model: Model,
    layout: Layout,
    data_init: Dict[str, int],
    rot_basis: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical initialization
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()
    for instruction in model.reset(qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    exc_qubits = set([q for q, s in data_init.items() if s])
    if exc_qubits:
        for instruction in model.x_gate(exc_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - exc_qubits
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits

    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit

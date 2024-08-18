from typing import List
import warnings
from itertools import compress

from stim import Circuit, target_rec
from qec_util import Layout

from ..models import Model


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

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_rounds = 1 if meas_reset else 2

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

    num_data, num_anc = len(data_qubits), len(anc_qubits)
    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    for anc_qubit in stab_qubits:
        neighbors = layout.get_neighbors(anc_qubit)
        neighbor_inds = layout.get_inds(neighbors)
        targets = [target_rec(ind - num_data) for ind in neighbor_inds]

        anc_ind = anc_qubits.index(anc_qubit)
        for round_ind in range(1, comp_rounds + 1):
            target = target_rec(anc_ind - num_data - round_ind * num_anc)
            targets.append(target)
        circuit.append("DETECTOR", targets)

    log_op = "log_x" if rot_basis else "log_z"
    if log_op not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_x and log_z in your layout.",
            DeprecationWarning,
        )
        targets = [target_rec(ind) for ind in range(-num_data, 0)]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    else:
        log_data_qubits = getattr(layout, log_op)
        targets = [target_rec(data_qubits.index(q) - num_data) for q in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def init_qubits(
    model: Model,
    layout: Layout,
    data_init: List[int],
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

    exc_qubits = set(compress(data_qubits, data_init))
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


def log_x(model: Model, layout: Layout) -> Circuit:
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

    return circuit


def log_z(model: Model, layout: Layout) -> Circuit:
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

    return circuit


def log_meas_xzzx(
    model: Model,
    layout: Layout,
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

    qubits = set(data_qubits + anc_qubits)

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_rounds = 1 if meas_reset else 2

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

    num_data, num_anc = len(data_qubits), len(anc_qubits)
    for anc_qubit in stab_qubits:
        neighbors = layout.get_neighbors(anc_qubit)
        neighbor_inds = layout.get_inds(neighbors)
        targets = [target_rec(ind - num_data) for ind in neighbor_inds]

        anc_ind = anc_qubits.index(anc_qubit)
        for round_ind in range(1, comp_rounds + 1):
            target = target_rec(anc_ind - num_data - round_ind * num_anc)
            targets.append(target)
        circuit.append("DETECTOR", targets)

    log_op = "log_x" if rot_basis else "log_z"
    if log_op not in dir(layout):
        warnings.warn(
            "Deprecation warning: specify log_x and log_z in your layout.",
            DeprecationWarning,
        )
        targets = [target_rec(ind) for ind in range(-num_data, 0)]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)
    else:
        log_data_qubits = getattr(layout, log_op)
        targets = [target_rec(data_qubits.index(q) - num_data) for q in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def init_qubits_xzzx(
    model: Model,
    layout: Layout,
    data_init: List[int],
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

    exc_qubits = set(compress(data_qubits, data_init))
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

from itertools import chain

from stim import Circuit
from qec_util import Layout

from ..models import Model


# methods to have in this script
from .util import qubit_coords, log_x, log_z
from .util import log_meas_xzzx as log_meas, init_qubits_xzzx as init_qubits

__all__ = ["qubit_coords", "log_meas", "log_x", "log_z", "qec_round", "init_qubits"]


def qec_round(
    model: Model,
    layout: Layout,
    meas_reset: bool = False,
    meas_comparison: bool = True,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.

    Params
    -------
    meas_comparison
        If True, the detector is set to the measurement of the ancilla
        instead of to the comparison of consecutive syndromes.
    stab_type_det
        If specified, only adds detectors to the ancillas for the
        specific stabilizator type.
    """
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    qubits = set(data_qubits + anc_qubits)

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_round = 1 if meas_reset else 2

    circuit = Circuit()
    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

        rot_qubits = set(stab_qubits)
        for direction in ("north_west", "south_east"):
            neighbors = layout.get_neighbors(stab_qubits, direction=direction)
            rot_qubits.update(neighbors)

        if not ind:
            for instruction in model.hadamard(rot_qubits):
                circuit.append(instruction)

            idle_qubits = qubits - rot_qubits
            for instruction in model.idle(idle_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        for ord_dir in int_order[stab_type]:
            int_pairs = layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))

            for instruction in model.cphase(int_qubits):
                circuit.append(instruction)

            idle_qubits = qubits - set(int_qubits)
            for instruction in model.idle(idle_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        if not ind:
            for instruction in model.hadamard(qubits):
                circuit.append(instruction)
        else:
            for instruction in model.hadamard(rot_qubits):
                circuit.append(instruction)

            idle_qubits = qubits - rot_qubits
            for instruction in model.idle(idle_qubits):
                circuit.append(instruction)

        circuit.append("TICK")

    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    for instruction in model.idle(data_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    if meas_reset:
        for instruction in model.reset(anc_qubits):
            circuit.append(instruction)

        for instruction in model.idle(data_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    # detectors ordered as in the measurements
    if meas_comparison:
        det_targets = []
        for qubit in anc_qubits:
            targets = [model.meas_target(qubit, -1), model.meas_target(qubit, -2)]
            det_targets.append(targets)
    else:
        det_targets = []
        for qubit in anc_qubits:
            targets = [model.meas_target(qubit, -1)]
            det_targets.append(targets)

    for targets in det_targets:
        circuit.append("DETECTOR", targets)

    return circuit

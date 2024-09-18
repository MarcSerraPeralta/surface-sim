from itertools import chain

from stim import Circuit

from ..layouts import Layout
from ..models import Model
from ..detectors import Detectors

# methods to have in this script
from .util import qubit_coords, log_meas, log_x, log_z, init_qubits, log_trans_s

__all__ = [
    "qubit_coords",
    "log_meas",
    "log_x",
    "log_z",
    "qec_round",
    "init_qubits",
    "log_trans_s",
]


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    meas_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.
    """
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()
    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        rot_qubits = set(stab_qubits)
        if stab_type == "x_type":
            rot_qubits.update(data_qubits)

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

    # add detectors
    detectors_stim = detectors.build_from_anc(model.meas_target, meas_reset)
    circuit += detectors_stim

    return circuit

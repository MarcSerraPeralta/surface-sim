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
    anc_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.
    """
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    circuit = Circuit()

    if anc_reset:
        circuit += model.reset(anc_qubits)
        circuit += model.idle(data_qubits)
        circuit += model.tick()

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        rot_qubits = set(stab_qubits)
        if stab_type == "x_type":
            rot_qubits.update(data_qubits)

        if not ind:
            idle_qubits = qubits - rot_qubits
            circuit += model.hadamard(rot_qubits)
            circuit += model.idle(idle_qubits)
            circuit += model.tick()

        for ord_dir in int_order[stab_type]:
            int_pairs = layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))
            idle_qubits = qubits - set(int_qubits)

            circuit += model.cphase(int_qubits)
            circuit += model.idle(idle_qubits)
            circuit += model.tick()

        if not ind:
            circuit += model.hadamard(qubits)
        else:
            idle_qubits = qubits - rot_qubits
            circuit += model.hadamard(rot_qubits)
            circuit += model.idle(idle_qubits)

        circuit += model.tick()

    circuit += model.measure(anc_qubits)
    circuit += model.idle(data_qubits)
    circuit += model.tick()

    # add detectors
    detectors_stim = detectors.build_from_anc(model.meas_target, anc_reset)
    circuit += detectors_stim

    return circuit

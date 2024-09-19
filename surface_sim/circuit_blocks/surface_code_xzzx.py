from itertools import chain

from stim import Circuit

from ..layouts import Layout
from ..models import Model
from ..detectors import Detectors


# methods to have in this script
from .util import qubit_coords, log_x, log_z
from .util import log_meas_xzzx as log_meas, init_qubits_xzzx as init_qubits

__all__ = ["qubit_coords", "log_meas", "log_x", "log_z", "qec_round", "init_qubits"]


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
        for direction in ("north_west", "south_east"):
            neighbors = layout.get_neighbors(stab_qubits, direction=direction)
            rot_qubits.update(neighbors)

        if not ind:
            circuit += model.hadamard(rot_qubits)

            idle_qubits = qubits - rot_qubits
            circuit += model.idle(idle_qubits)
            circuit += model.tick()

        for ord_dir in int_order[stab_type]:
            int_pairs = layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))

            circuit += model.cphase(int_qubits)

            idle_qubits = qubits - set(int_qubits)
            circuit += model.idle(idle_qubits)
            circuit += model.tick()

        if not ind:
            circuit += model.hadamard(qubits)
        else:
            circuit += model.hadamard(rot_qubits)

            idle_qubits = qubits - rot_qubits
            circuit += model.idle(idle_qubits)

        circuit += model.tick()

    circuit += model.measure(anc_qubits)

    circuit += model.idle(data_qubits)
    circuit += model.tick()

    if meas_reset:
        circuit += model.reset(anc_qubits)

        circuit += model.idle(data_qubits)

        circuit += model.tick()

    # add detectors
    detectors_stim = detectors.build_from_anc(model.meas_target, meas_reset)
    circuit += detectors_stim

    return circuit

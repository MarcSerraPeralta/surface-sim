from itertools import chain

from stim import Circuit

from ..layouts import Layout
from ..models import Model
from ..detectors import Detectors

# methods to have in this script
from .util import qubit_coords
from .util import log_x_xzzx as log_x
from .util import log_z_xzzx as log_z
from .util import log_meas_xzzx as log_meas
from .util import init_qubits_xzzx as init_qubits

__all__ = [
    "qubit_coords",
    "log_meas",
    "log_x",
    "log_z",
    "qec_round",
    "init_qubits",
]


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    anc_reset
        If True, ancillas are reset at the beginning of the QEC cycle.
        By default True.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.

    Notes
    -----
    This implementation follows:

    https://doi.org/10.1103/PhysRevApplied.8.034021
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )
    if anc_detectors is None:
        anc_detectors = layout.get_qubits(role="anc")
    if set(anc_detectors) > set(layout.get_qubits(role="anc")):
        raise ValueError("Some of the given 'anc_qubits' are not ancilla qubits.")

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())
    x_stabs = layout.get_qubits(role="anc", stab_type="x_type")

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    if anc_reset:
        circuit += model.reset(anc_qubits)
        circuit += model.idle(data_qubits)
        circuit += model.tick()

    # a
    directions = [int_order["x_type"][0], int_order["x_type"][3]]
    rot_qubits = set(anc_qubits)
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[0]))
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[1]))
    rot_qubits_xzzx = set()
    for direction in ("north_west", "south_east"):
        stab_qubits = layout.get_qubits(role="anc", stab_type="z_type")
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits_xzzx.update(neighbors)
    rot_qubits.symmetric_difference_update(rot_qubits_xzzx)
    idle_qubits = qubits - rot_qubits

    circuit += model.hadamard(rot_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # b
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][0]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # c
    circuit += model.hadamard(data_qubits)
    circuit += model.idle(anc_qubits)
    circuit += model.tick()

    # d
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][1]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # e
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][2]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # f
    circuit += model.hadamard(data_qubits)
    circuit += model.idle(anc_qubits)
    circuit += model.tick()

    # g
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][3]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # h
    directions = [int_order["x_type"][0], int_order["x_type"][3]]
    rot_qubits = set(anc_qubits)
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[0]))
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[1]))
    rot_qubits_xzzx = set()
    for direction in ("north_west", "south_east"):
        stab_qubits = layout.get_qubits(role="anc", stab_type="z_type")
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits_xzzx.update(neighbors)
    rot_qubits.symmetric_difference_update(rot_qubits_xzzx)
    idle_qubits = qubits - rot_qubits

    circuit += model.hadamard(rot_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # i
    circuit += model.measure(anc_qubits)
    circuit += model.idle(data_qubits)
    circuit += model.tick()

    # add detectors
    detectors_stim = detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )
    circuit += detectors_stim

    return circuit

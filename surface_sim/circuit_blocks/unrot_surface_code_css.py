from collections.abc import Generator, Collection
from itertools import chain

from stim import Circuit

from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from .decorators import qec_circuit

# methods to have in this script
from .util import (
    qubit_coords,
    idle_iterator,
    log_meas,
    log_meas_iterator,
    log_meas_z_iterator,
    log_meas_x_iterator,
    log_x,
    log_x_iterator,
    log_z,
    log_z_iterator,
    init_qubits,
    init_qubits_iterator,
    init_qubits_z0_iterator,
    init_qubits_z1_iterator,
    init_qubits_x0_iterator,
    init_qubits_x1_iterator,
    log_fold_trans_s,
    log_fold_trans_s_iterator,
    log_fold_trans_h,
    log_fold_trans_h_iterator,
    log_trans_cnot,
    log_trans_cnot_iterator,
    qec_round_iterator,
    qec_round_iterator_cnots,
)

__all__ = [
    "qubit_coords",
    "idle_iterator",
    "log_meas",
    "log_meas_iterator",
    "log_meas_z_iterator",
    "log_meas_x_iterator",
    "log_x",
    "log_x_iterator",
    "log_z",
    "log_z_iterator",
    "init_qubits",
    "init_qubits_iterator",
    "init_qubits_z0_iterator",
    "init_qubits_z1_iterator",
    "init_qubits_x0_iterator",
    "init_qubits_x1_iterator",
    "log_fold_trans_s",
    "log_fold_trans_s_iterator",
    "log_fold_trans_h",
    "log_fold_trans_h_iterator",
    "log_trans_cnot",
    "log_trans_cnot_iterator",
    "qec_round",
    "qec_round_iterator",
    "qec_round_iterator_cnots",
    "gate_to_iterator",
    "gate_to_iterator_cnots",
]


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC round
    of the given model.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector object to use for their definition.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.

    Notes
    -----
    This implementation follows:

    https://doi.org/10.1103/PhysRevApplied.8.034021
    """
    circuit = sum(
        qec_round_iterator(model=model, layout=layout, anc_reset=anc_reset),
        start=Circuit(),
    )

    # add detectors
    anc_qubits = layout.anc_qubits
    if anc_detectors is None:
        anc_detectors = anc_qubits
    if set(anc_detectors) > set(anc_qubits):
        raise ValueError("Elements in 'anc_detectors' are not ancilla qubits.")

    circuit += detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )

    return circuit


@qec_circuit
def qec_round_iterator(
    model: Model,
    layout: Layout,
    anc_reset: bool = True,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to a QEC round
    of the given model without the detectors.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
        By default ``True``.

    Notes
    -----
    This implementation follows:

    https://doi.org/10.1103/PhysRevApplied.8.034021
    """
    if layout.code != "unrotated_surface_code":
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    qubits = set(layout.qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())
    x_stabs = layout.get_qubits(role="anc", stab_type="x_type")

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        yield model.reset(anc_qubits) + model.idle(data_qubits)
        yield model.tick()

    # a
    directions = [int_order["x_type"][0], int_order["x_type"][3]]
    rot_qubits = set(anc_qubits)
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[0]))
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[1]))
    idle_qubits = qubits - rot_qubits

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # b
    int_qubits: list[str] = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][0]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # c
    yield model.hadamard(data_qubits) + model.idle(anc_qubits)
    yield model.tick()

    # d
    int_qubits = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][1]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # e
    int_qubits = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][2]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # f
    yield model.hadamard(data_qubits) + model.idle(anc_qubits)
    yield model.tick()

    # g
    int_qubits = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][3]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # h
    directions = [int_order["x_type"][0], int_order["x_type"][3]]
    rot_qubits = set(anc_qubits)
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[0]))
    rot_qubits.update(layout.get_neighbors(x_stabs, direction=directions[1]))
    idle_qubits = qubits - rot_qubits

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # i
    yield model.measure(anc_qubits) + model.idle(data_qubits)
    yield model.tick()


@qec_circuit
def qec_round_iterator_cnots(
    model: Model,
    layout: Layout,
    anc_reset: bool = True,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to a QEC round
    of the given model without the detectors.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC round.
        By default ``True``.

    Notes
    -----
    This implementation uses the following instructions: CNOT, RZ, RX, MZ, MX.
    Note that if ``anc_reset = False``, then the ancillas are not reset in the first round
    and stim assumes that, if not specified, they are reset in the Z-basis, which is the
    incorrect basis for the X-type ancillas. See the initialization iterators from the
    dodecahedron code.
    """
    if layout.code != "unrotated_surface_code":
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    x_stabs = layout.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout.get_qubits(role="anc", stab_type="z_type")
    qubits = set(layout.qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        resets = model.reset_x(x_stabs) + model.reset_z(z_stabs)
        yield resets + model.idle(data_qubits)
        yield model.tick()

    # a
    int_qubits: list[str] = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][0]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        if stab_type == "z_type":
            int_pairs = [(b, a) for (a, b) in int_pairs]
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cnot(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # b
    int_qubits = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][1]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        if stab_type == "z_type":
            int_pairs = [(b, a) for (a, b) in int_pairs]
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cnot(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # c
    int_qubits = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][2]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        if stab_type == "z_type":
            int_pairs = [(b, a) for (a, b) in int_pairs]
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cnot(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # d
    int_qubits = []
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][3]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        if stab_type == "z_type":
            int_pairs = [(b, a) for (a, b) in int_pairs]
        int_qubits += list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cnot(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # e
    yield model.measure_x(x_stabs) + model.measure_z(z_stabs) + model.idle(data_qubits)
    yield model.tick()


gate_to_iterator = {
    "TICK": qec_round_iterator,
    "I": idle_iterator,
    "S": log_fold_trans_s_iterator,
    "H": log_fold_trans_h_iterator,
    "X": log_x_iterator,
    "Z": log_z_iterator,
    "CX": log_trans_cnot_iterator,
    "CNOT": log_trans_cnot_iterator,
    "R": init_qubits_z0_iterator,
    "RZ": init_qubits_z0_iterator,
    "RX": init_qubits_x0_iterator,
    "M": log_meas_z_iterator,
    "MZ": log_meas_z_iterator,
    "MX": log_meas_x_iterator,
}
gate_to_iterator_cnots = {
    "TICK": qec_round_iterator_cnots,
    "I": idle_iterator,
    "S": log_fold_trans_s_iterator,
    "H": log_fold_trans_h_iterator,
    "X": log_x_iterator,
    "Z": log_z_iterator,
    "CX": log_trans_cnot_iterator,
    "CNOT": log_trans_cnot_iterator,
    "R": init_qubits_z0_iterator,
    "RZ": init_qubits_z0_iterator,
    "RX": init_qubits_x0_iterator,
    "M": log_meas_z_iterator,
    "MZ": log_meas_z_iterator,
    "MX": log_meas_x_iterator,
}

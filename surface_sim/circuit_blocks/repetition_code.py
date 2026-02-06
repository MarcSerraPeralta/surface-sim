from collections.abc import Collection, Generator
from itertools import chain

from stim import Circuit

from ..detectors import Detectors
from ..layouts.layout import Layout
from ..models import Model
from .decorators import qec_circuit

# methods to have in this script
from .util import (
    idle_iterator,
    init_qubits,
    init_qubits_iterator,
    init_qubits_x0_iterator,
    init_qubits_x1_iterator,
    init_qubits_z0_iterator,
    init_qubits_z1_iterator,
    log_meas,
    log_meas_iterator,
    log_meas_x_iterator,
    log_meas_z_iterator,
    log_x,
    log_x_iterator,
    log_z,
    log_z_iterator,
    qubit_coords,
)
from .util import qec_round_iterator as qec_round_iterator_sc

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
    "qec_round",
    "qec_round_iterator",
    "gate_to_iterator",
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
    This implementation follows two schedules, depending on the layout's interaction order:

    https://doi.org/10.1103/PhysRevApplied.8.034021

    or

    shallowest interaction order for the rotated surface code where ancillas interact
    with the data qubit on the left and right in just two two-qubit-gate layers.

    It activates all the ancillas in ``detectors`` to always build the detectors.
    As this function should not be used when building encoded circuits with
    the iterating functions, it does not matter if the detectors are activated or not.
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

    # activate detectors so that "Detectors.build_from_anc" always populates
    # the stim detector definitions.
    inactive_dets = set(anc_detectors).difference(detectors.detectors)
    detectors.activate_detectors(inactive_dets)

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
    This implementation follows two schedules, depending on the layout's interaction order:

    https://doi.org/10.1103/PhysRevApplied.8.034021

    or

    shallowest interaction order for the rotated surface code where ancillas interact
    with the data qubit on the left and right in just two two-qubit-gate layers.
    """
    if layout.code not in ("repetition_code", "repetition_stability"):
        raise TypeError(
            f"The given layout is not a repetition code, but a {layout.code}"
        )
    if "steps" not in layout.interaction_order:
        yield from qec_round_iterator_sc(model, layout, anc_reset=anc_reset)
        return  # finish generator

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    qubits = set(layout.qubits)

    int_order = layout.interaction_order.copy()
    mapping = {
        "left": ("north_west", "south_west"),
        "right": ("north_east", "south_east"),
    }
    int_order["steps"] = [mapping[i] for i in int_order["steps"]]

    stab_type = (
        "z_type"
        if len(layout.get_qubits(role="anc", stab_type="x_type")) == 0
        else "x_type"
    )

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        yield model.reset(anc_qubits) + model.idle(data_qubits)
        yield model.tick()

    # a
    rot_qubits = set(anc_qubits)
    if stab_type == "x_type":
        rot_qubits.update(data_qubits)
    idle_qubits = qubits - rot_qubits

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # b
    int_pairs: list[tuple[str, str]] = []
    for ord_dir in int_order["steps"][0]:
        int_pairs.extend(
            layout.get_neighbors(anc_qubits, direction=ord_dir, as_pairs=True)
        )
    int_qubits = list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # c
    int_pairs = []
    for ord_dir in int_order["steps"][1]:
        int_pairs += layout.get_neighbors(anc_qubits, direction=ord_dir, as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # d
    rot_qubits = set(anc_qubits)
    if stab_type == "x_type":
        rot_qubits.update(data_qubits)
    idle_qubits = qubits - rot_qubits

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # e
    yield model.measure(anc_qubits) + model.idle(data_qubits)
    yield model.tick()


gate_to_iterator = {
    "TICK": qec_round_iterator,
    "I": idle_iterator,
    "X": log_x_iterator,
    "Z": log_z_iterator,
    "R": init_qubits_z0_iterator,
    "RZ": init_qubits_z0_iterator,
    "RX": init_qubits_x0_iterator,
    "M": log_meas_z_iterator,
    "MZ": log_meas_z_iterator,
    "MX": log_meas_x_iterator,
}

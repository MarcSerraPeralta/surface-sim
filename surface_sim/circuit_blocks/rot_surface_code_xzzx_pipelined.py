from collections.abc import Iterator, Sequence
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
from .util import log_meas_xzzx_iterator as log_meas_iterator
from .util import init_qubits_xzzx as init_qubits

__all__ = [
    "qubit_coords",
    "log_meas",
    "log_meas_iterator",
    "log_x",
    "log_z",
    "qec_round",
    "qec_round_iterator",
    "init_qubits",
]


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
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
        Detector object to use for their definition.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
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
    anc_qubits = layout.get_qubits(role="anc")
    if anc_detectors is None:
        anc_detectors = anc_qubits
    if set(anc_detectors) > set(anc_qubits):
        raise ValueError("Elements in 'anc_detectors' are not ancilla qubits.")

    circuit += detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )

    return circuit


def qec_round_iterator(
    model: Model,
    layout: Layout,
    anc_reset: bool = True,
) -> Iterator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to a QEC cycle
    of the given model without the detectors.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        yield model.reset(anc_qubits)
        yield model.idle(data_qubits)
        yield model.tick()

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

        rot_qubits = set(stab_qubits)
        for direction in ("north_west", "south_east"):
            neighbors = layout.get_neighbors(stab_qubits, direction=direction)
            rot_qubits.update(neighbors)

        if not ind:
            idle_qubits = qubits - rot_qubits
            yield model.hadamard(rot_qubits)
            yield model.idle(idle_qubits)
            yield model.tick()

        for ord_dir in int_order[stab_type]:
            int_pairs = layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))
            idle_qubits = qubits - set(int_qubits)

            yield model.cphase(int_qubits)
            yield model.idle(idle_qubits)
            yield model.tick()

        if not ind:
            yield model.hadamard(qubits)
            yield model.tick()
        else:
            idle_qubits = qubits - rot_qubits
            yield model.hadamard(rot_qubits)
            yield model.idle(idle_qubits)
            yield model.tick()

    yield model.measure(anc_qubits)
    yield model.idle(data_qubits)
    yield model.tick()

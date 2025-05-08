from collections.abc import Iterator, Sequence
from itertools import chain

from stim import Circuit

from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from .decorators import qec_circuit

# methods to have in this script
from .util import qubit_coords, idle_iterator
from .util import log_x_xzzx as log_x
from .util import log_x_xzzx_iterator as log_x_iterator
from .util import log_z_xzzx as log_z
from .util import log_z_xzzx_iterator as log_z_iterator
from .util import log_meas_xzzx as log_meas
from .util import log_meas_xzzx_iterator as log_meas_iterator
from .util import log_meas_z_xzzx_iterator as log_meas_z_iterator
from .util import log_meas_x_xzzx_iterator as log_meas_x_iterator
from .util import init_qubits_xzzx as init_qubits
from .util import init_qubits_xzzx_iterator as init_qubits_iterator
from .util import init_qubits_z0_xzzx_iterator as init_qubits_z0_iterator
from .util import init_qubits_z1_xzzx_iterator as init_qubits_z1_iterator
from .util import init_qubits_x0_xzzx_iterator as init_qubits_x0_iterator
from .util import init_qubits_x1_xzzx_iterator as init_qubits_x1_iterator

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
]


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
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
) -> Iterator[Circuit]:
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
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
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
        yield model.reset(anc_qubits) + model.idle_reset(data_qubits)
        yield model.tick()

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

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # b
    cz_circuit = Circuit()
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][0]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        cz_circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    yield cz_circuit + model.idle(idle_qubits)
    yield model.tick()

    # c
    yield model.hadamard(data_qubits) + model.idle(anc_qubits)
    yield model.tick()

    # d
    cz_circuit = Circuit()
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][1]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        cz_circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    yield cz_circuit + model.idle(idle_qubits)
    yield model.tick()

    # e
    cz_circuit = Circuit()
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][2]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        cz_circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    yield cz_circuit + model.idle(idle_qubits)
    yield model.tick()

    # f
    yield model.hadamard(data_qubits) + model.idle(anc_qubits)
    yield model.tick()

    # g
    cz_circuit = Circuit()
    interacted_qubits = set()
    for stab_type in stab_types:
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        ord_dir = int_order[stab_type][3]
        int_pairs = layout.get_neighbors(stab_qubits, direction=ord_dir, as_pairs=True)
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)

        cz_circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(interacted_qubits)
    yield cz_circuit + model.idle(idle_qubits)
    yield model.tick()

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

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # i
    yield model.measure(anc_qubits) + model.idle_meas(data_qubits)
    yield model.tick()


def qec_round_pipelined(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Sequence[str] | None = None,
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
        qec_round_pipelined_iterator(model=model, layout=layout, anc_reset=anc_reset),
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
def qec_round_pipelined_iterator(
    model: Model,
    layout: Layout,
    anc_reset: bool = True,
) -> Iterator[Circuit]:
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
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    qubits = set(layout.qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        yield model.reset(anc_qubits) + model.idle_reset(data_qubits)
        yield model.tick()

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

        rot_qubits = set(stab_qubits)
        for direction in ("north_west", "south_east"):
            neighbors = layout.get_neighbors(stab_qubits, direction=direction)
            rot_qubits.update(neighbors)

        if not ind:
            idle_qubits = qubits - rot_qubits
            yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
            yield model.tick()

        for ord_dir in int_order[stab_type]:
            int_pairs = layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))
            idle_qubits = qubits - set(int_qubits)

            yield model.cphase(int_qubits) + model.idle(idle_qubits)
            yield model.tick()

        if not ind:
            yield model.hadamard(qubits)
            yield model.tick()
        else:
            idle_qubits = qubits - rot_qubits
            yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
            yield model.tick()

    yield model.measure(anc_qubits) + model.idle_meas(data_qubits)
    yield model.tick()


def qec_round_google_with_log_meas(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_detectors: list[str] | None = None,
    rot_basis: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC round
    that includes the logical measurement
    of the given model. It defines the observables for
    all logical qubits in the layout.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.

    Notes
    -----
    The circuits are based on the following paper by Google AI:
    https://doi.org/10.1038/s41586-022-05434-1
    https://doi.org/10.48550/arXiv.2207.06431
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )
    if anc_detectors is None:
        anc_detectors = layout.anc_qubits
    if set(anc_detectors) > set(layout.anc_qubits):
        raise ValueError("Some of the given 'anc_qubits' are not ancilla qubits.")

    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)

    # a-h
    circuit = coherent_qec_part_google(model=model, layout=layout)

    # i (for logical measurement)
    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set(anc_qubits)
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    circuit += model.hadamard(rot_qubits)

    idle_qubits = qubits - rot_qubits
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # j (for logical measurement)
    circuit += model.measure(anc_qubits)
    circuit += model.measure(data_qubits)

    # detectors and logical observables
    detectors_stim = detectors.build_from_anc(
        model.meas_target, anc_reset=True, anc_qubits=anc_detectors
    )
    circuit += detectors_stim

    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    anc_support = layout.get_support(stabs)
    detectors_stim = detectors.build_from_data(
        model.meas_target,
        anc_support,
        anc_reset=True,
        reconstructable_stabs=stabs,
        anc_qubits=anc_detectors,
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    for logical_qubit in layout.logical_qubits:
        log_data_qubits = layout.logical_param(log_op, logical_qubit)
        targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    detectors.deactivate_detectors(layout.anc_qubits)

    return circuit


def coherent_qec_part_google(model: Model, layout: Layout) -> Circuit:
    """
    Returns stim circuit corresponding to the steps "a" to "h" from the QEC round
    described in Google's paper for the given model.

    Notes
    -----
    The circuits are based on the following paper by Google AI:
    https://doi.org/10.1038/s41586-022-05434-1
    https://doi.org/10.48550/arXiv.2207.06431
    """
    data_qubits = layout.data_qubits
    x_anc = layout.get_qubits(role="anc", stab_type="x_type")
    z_anc = layout.get_qubits(role="anc", stab_type="z_type")
    anc_qubits = x_anc + z_anc
    qubits = set(layout.qubits)

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    # a
    circuit += model.hadamard(anc_qubits)
    circuit += model.x_gate(data_qubits)
    circuit += model.tick()

    # b
    int_pairs = layout.get_neighbors(anc_qubits, direction="north_east", as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # c
    circuit += model.hadamard(data_qubits)
    circuit += model.x_gate(anc_qubits)
    circuit += model.tick()

    # d
    x_pairs = layout.get_neighbors(x_anc, direction="north_west", as_pairs=True)
    z_pairs = layout.get_neighbors(z_anc, direction="south_east", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # e
    circuit += model.x_gate(qubits)
    circuit += model.tick()

    # f
    x_pairs = layout.get_neighbors(x_anc, direction="south_east", as_pairs=True)
    z_pairs = layout.get_neighbors(z_anc, direction="north_west", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # g
    circuit += model.hadamard(data_qubits)
    circuit += model.x_gate(anc_qubits)
    circuit += model.tick()

    # h
    int_pairs = layout.get_neighbors(anc_qubits, direction="south_west", as_pairs=True)
    int_qubits = list(chain.from_iterable(int_pairs))

    circuit += model.cphase(int_qubits)

    idle_qubits = qubits - set(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    return circuit


def qec_round_google(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_detectors: list[str] | None = None,
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
        Detector definitions to use.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.

    Notes
    -----
    The circuits are based on the following paper by Google AI:
    https://doi.org/10.1038/s41586-022-05434-1
    https://doi.org/10.48550/arXiv.2207.06431
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            "The given layout is not a rotated surface code, " f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits

    # a-h
    circuit = coherent_qec_part_google(model=model, layout=layout)

    # i
    circuit += model.hadamard(anc_qubits)
    circuit += model.x_gate(data_qubits)
    circuit += model.tick()

    # j
    circuit += model.measure(anc_qubits)

    circuit += model.idle_meas(data_qubits)
    circuit += model.tick()

    circuit += model.reset(anc_qubits)
    circuit += model.idle_reset(data_qubits)
    circuit += model.tick()

    # add detectors
    detectors_stim = detectors.build_from_anc(
        model.meas_target, anc_reset=True, anc_qubits=anc_detectors
    )
    circuit += detectors_stim

    return circuit


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

from collections.abc import Generator, Collection, Callable
from itertools import chain

from stim import Circuit

from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from .decorators import qec_circuit, tq_gate, qubit_encoding

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
    log_trans_cnot,
    log_trans_cnot_iterator,
    qec_round_iterator,
    qec_round_iterator_cnots,
    to_mid_cycle_iterator_cnots,
    to_end_cycle_iterator_cnots,
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
    "log_trans_cnot",
    "log_trans_cnot_iterator",
    "qec_round",
    "qec_round_iterator",
    "qec_round_iterator_cnots",
    "qec_round_pipelined",
    "qec_round_pipelined_iterator",
    "log_trans_cnot_mid_cycle_css_iterator",
    "encoding_qubits_d3_iterator_cnots",
    "encoding_qubits_x0_d3_iterator_cnots",
    "encoding_qubits_y0_d3_iterator_cnots",
    "encoding_qubits_z0_d3_iterator_cnots",
    "gate_to_iterator",
    "gate_to_iterator_cnots",
    "gate_to_iterator_mid_cycle_cnots",
    "tick_iterators_mid_cycle_cnots",
    "gate_to_iterator_pipelined",
]


def qec_round(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC round of the given model.

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


def qec_round_pipelined(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    anc_reset: bool = True,
    anc_detectors: Collection[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC round of the given model.

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

    It activates all the ancillas in ``detectors`` to always build the detectors.
    As this function should not be used when building encoded circuits with
    the iterating functions, it does not matter if the detectors are activated or not.
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

    # activate detectors so that "Detectors.build_from_anc" always populates
    # the stim detector definitions.
    inactive_dets = set(anc_detectors).difference(detectors.detectors)
    detectors.activate_detectors(inactive_dets)

    circuit += detectors.build_from_anc(
        model.meas_target, anc_reset, anc_qubits=anc_detectors
    )

    return circuit


@qec_circuit
def qec_round_pipelined_iterator(
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
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    qubits = set(layout.qubits)

    int_order = layout.interaction_order
    stab_types = list(int_order.keys())

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        yield model.reset(anc_qubits) + model.idle(data_qubits)
        yield model.tick()

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        rot_qubits = set(stab_qubits)
        if stab_type == "x_type":
            rot_qubits.update(data_qubits)

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
        else:
            idle_qubits = qubits - rot_qubits
            yield model.hadamard(rot_qubits) + model.idle(idle_qubits)

        yield model.tick()

    yield model.measure(anc_qubits) + model.idle(data_qubits)
    yield model.tick()


@tq_gate
def log_trans_cnot_mid_cycle_css_iterator(
    model: Model, layout_c: Layout, layout_t: Layout
) -> Generator[Circuit]:
    """Returns the stim circuit corresponding to a transversal logical CNOT gate.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout_c
        Code layout for the control of the logical CNOT.
    layout_t
        Code layout for the target of the logical CNOT.

    Notes
    -----
    The implementation uses CNOTs and it does not add incoming noise because this
    logical gate is executed in the mid-cycle code.
    """
    if layout_c.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout_c.code}"
        )
    if layout_t.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout_t.code}"
        )

    qubits = set(layout_c.qubits + layout_t.qubits)
    gate_label = f"log_trans_cnot_mid_cycle_css_{layout_c.logical_qubits[0]}_{layout_t.logical_qubits[0]}"

    cnot_pairs: set[tuple[str, str]] = set()
    for qubit in layout_c.qubits:
        trans_cnot = layout_c.param(gate_label, qubit)
        if trans_cnot is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        cnot_pairs.add((qubit, trans_cnot["cnot"]))

    # long-range CNOT gates
    int_qubits = list(chain.from_iterable(cnot_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.cnot(int_qubits) + model.idle(idle_qubits)
    yield model.tick()


def encoding_qubits_iterator_cnots(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to a rotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    if layout.distance % 2 == 1:
        yield from encoding_qubits_odd_d_iterator_cnots(
            model=model, layout=layout, physical_reset_op=physical_reset_op
        )
    else:
        yield from encoding_qubits_even_d_iterator_cnots(
            model=model, layout=layout, physical_reset_op=physical_reset_op
        )


def encoding_qubits_even_d_iterator_cnots(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to a even-distance rotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}."
        )
    if layout.distance % 2 != 0:
        raise TypeError(
            f"The given layout does not have even distance, but distance {layout.distance}."
        )

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    l: dict[tuple[int, int], str] = {}
    for qubit in layout.data_qubits:
        glabel = layout.param(gate_label, qubit)["label"]
        if glabel is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        l[glabel] = qubit

    # first build d=2 rotated surface code.
    # the CNOTs and resets depend wether the weight-2 stabilizers are Z-type or X-type.
    z_anc = layout.get_neighbors([l[(0, 0)]], stab_type="z_type")[0]
    reversed = len(layout.get_neighbors([z_anc])) != 2

    # step 1
    circ = Circuit()
    reset_z, reset_x = [l[(0, 2)], l[(0, 3)]], [l[(0, 1)]]
    if reversed:
        reset_z, reset_x = reset_x, reset_z
    circ += model.reset_z(reset_z)
    circ += model.reset_x(reset_x)
    circ += model.__getattribute__(physical_reset_op)([l[(0, 0)]])
    yield circ + model.idle(layout.anc_qubits)
    yield model.tick()

    # step 2
    pair1 = l[(0, 0)], l[(0, 3)]
    pair2 = l[(0, 1)], l[(0, 2)]
    if reversed:
        pair1 = pair1[::-1]
        pair2 = pair2[::-1]
    circ = Circuit()
    circ += model.cnot([*pair1, *pair2])
    circ += model.idle(layout.anc_qubits)
    yield circ
    yield model.tick()

    # step 3
    pair1 = l[(0, 1)], l[(0, 0)]
    pair2 = l[(0, 2)], l[(0, 3)]
    if reversed:
        pair1 = pair1[::-1]
        pair2 = pair2[::-1]
    circ = Circuit()
    circ += model.cnot([*pair1, *pair2])
    circ += model.idle(layout.anc_qubits)
    yield circ
    yield model.tick()

    # grow code to maximum size
    for d in range(2, layout.distance, 2):
        yield from grow_code_even_d_iterator_cnots(
            model=model, layout=layout, init_distance=d
        )


def grow_code_even_d_iterator_cnots(
    model: Model,
    layout: Layout,
    init_distance: int,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an circuit growing
    an even-distance ``d`` rotated surface code to a ``d+2`` one of the given model
    without the detectors. Note that this circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a rotated surface code.

    Notes
    -----
    The implementation follows Figure 2 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}."
        )
    if layout.distance % 2 != 0:
        raise TypeError(
            f"The given layout does not have even distance, but distance {layout.distance}."
        )
    if init_distance % 2 != 0:
        raise TypeError(f"'init_distance' must be even, but {init_distance} was given.")
    if init_distance >= layout.distance:
        raise TypeError(
            f"'init_distance' ({init_distance}) must be strictly larger than "
            f"the code distance ({layout.distance})."
        )

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    l: dict[tuple[int, int], str] = {}
    for qubit in layout.data_qubits:
        glabel = layout.param(gate_label, qubit)["label"]
        if glabel is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        l[glabel] = qubit

    inner_ring = (init_distance - 1) // 2
    outer_ring = inner_ring + 1
    length = init_distance + 1
    qubits = set(layout.data_qubits)

    # the CNOTs and resets depend wether the weight-2 stabilizers are Z-type or X-type.
    z_anc = layout.get_neighbors([l[(0, 0)]], stab_type="z_type")[0]
    reversed = len(layout.get_neighbors([z_anc])) != 2

    # step 1
    reset_x: list[str] = []
    reset_z: list[str] = [l[(outer_ring, 0 + i * length)] for i in range(4)]
    for i in [1, 3]:
        reset_x.append(l[(outer_ring, 1 + i * length)])
        reset_x.append(l[(outer_ring, length - 1 + i * length)])

    reset_z += [l[(outer_ring, p + 0 * length)] for p in range(1, length - 1, 2)]
    reset_x += [l[(outer_ring, p + 0 * length)] for p in range(2, length, 2)]

    reset_x += [l[(outer_ring, p + 2 * length)] for p in range(1, length - 1, 2)]
    reset_z += [l[(outer_ring, p + 2 * length)] for p in range(2, length, 2)]

    reset_x += [l[(outer_ring, p + 1 * length)] for p in range(2, length - 2, 2)]
    reset_z += [l[(outer_ring, p + 1 * length)] for p in range(3, length - 1, 2)]

    reset_z += [l[(outer_ring, p + 3 * length)] for p in range(2, length - 2, 2)]
    reset_x += [l[(outer_ring, p + 3 * length)] for p in range(3, length - 1, 2)]

    if reversed:
        reset_x, reset_z = reset_z, reset_x
    idling = qubits - set(reset_x) - set(reset_z)

    yield model.reset_x(reset_x) + model.reset_z(reset_z) + model.idle(idling)
    yield model.tick()

    # step 2
    cnots: list[str] = []

    odd = [l[(outer_ring, p + 0 * length)] for p in range(1, length - 1, 2)]
    even = [l[(outer_ring, p + 0 * length)] for p in range(2, length, 2)]
    if reversed:
        odd, even = even, odd
    cnots += list(chain.from_iterable(zip(even, odd)))

    odd = [l[(outer_ring, p + 2 * length)] for p in range(1, length - 1, 2)]
    even = [l[(outer_ring, p + 2 * length)] for p in range(2, length, 2)]
    if reversed:
        odd, even = even, odd
    cnots += list(chain.from_iterable(zip(odd, even)))

    even = [l[(outer_ring, p + 1 * length)] for p in range(2, length - 2, 2)]
    odd = [l[(outer_ring, p + 1 * length)] for p in range(3, length - 1, 2)]
    if reversed:
        odd, even = even, odd
    cnots += list(chain.from_iterable(zip(even, odd)))

    outer = [l[(outer_ring, 1 + 1 * length)], l[(outer_ring, length - 1 + 1 * length)]]
    inner = [
        l[(inner_ring, 0 + 1 * (length - 2))],
        l[(inner_ring, 0 + 2 * (length - 2))],
    ]
    if reversed:
        inner, outer = outer, inner
    cnots += list(chain.from_iterable(zip(outer, inner)))

    even = [l[(outer_ring, p + 3 * length)] for p in range(2, length - 2, 2)]
    odd = [l[(outer_ring, p + 3 * length)] for p in range(3, length - 1, 2)]
    if reversed:
        odd, even = even, odd
    cnots += list(chain.from_iterable(zip(odd, even)))

    outer = [l[(outer_ring, 1 + 3 * length)], l[(outer_ring, length - 1 + 3 * length)]]
    inner = [l[(inner_ring, 0 + 3 * (length - 2))], l[(inner_ring, 0)]]
    if reversed:
        inner, outer = outer, inner
    cnots += list(chain.from_iterable(zip(outer, inner)))

    yield model.cnot(cnots) + model.idle(idling)
    yield model.tick()

    # step 3
    cnots = []
    for i in [0, 2]:
        first = l[(outer_ring, 0 + length * i)]
        last = l[(outer_ring, (length - 1 + length * ((i - 1) % 4)))]
        if reversed:
            first, last = last, first
        cnots += [last, first]

        outer = [l[(outer_ring, p + i * length)] for p in range(1, length)]
        inner = [l[(inner_ring, p + i * (length - 2))] for p in range(length - 2)]
        inner.append(l[(inner_ring, 0 + (i + 1) * (length - 2))])
        if reversed:
            inner, outer = outer, inner
        cnots += list(chain.from_iterable(zip(inner, outer)))

    for i in [1, 3]:
        first = l[(outer_ring, 0 + length * i)]
        second = l[(outer_ring, 1 + length * i)]
        if reversed:
            first, second = second, first
        cnots += [second, first]

        outer = [l[(outer_ring, p + i * length)] for p in range(2, length - 1)]
        inner = [l[(inner_ring, p + i * (length - 2))] for p in range(1, length - 2)]
        if reversed:
            inner, outer = outer, inner
        cnots += list(chain.from_iterable(zip(outer, inner)))

    yield model.cnot(cnots) + model.idle(idling)
    yield model.tick()


def encoding_qubits_odd_d_iterator_cnots(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to a odd-distance rotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}."
        )
    if layout.distance % 2 == 0:
        raise TypeError(
            f"The given layout does not have odd distance, but distance {layout.distance}."
        )

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    l: dict[tuple[int, int], str] = {}
    for qubit in layout.data_qubits:
        glabel = layout.param(gate_label, qubit)["label"]
        if glabel is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        l[glabel] = qubit

    # first build d=3 rotated surface code
    # step 1
    circ = Circuit()
    circ += model.reset_z([l[(1, 0)], l[(1, 2)], l[(1, 6)], l[(1, 4)]])
    circ += model.reset_x([l[(1, 1)], l[(1, 7)], l[(1, 3)], l[(1, 5)]])
    circ += model.__getattribute__(physical_reset_op)([l[(0, 0)]])
    yield circ + model.idle(layout.anc_qubits)
    yield model.tick()

    # step 2
    circ = Circuit()
    circ += model.cnot(
        [l[(1, 1)], l[(1, 2)], l[(1, 3)], l[(0, 0)], l[(1, 5)], l[(1, 6)]]
    )
    circ += model.idle([l[(1, 0)], l[(1, 7)], l[(1, 4)], *layout.anc_qubits])
    yield circ
    yield model.tick()

    # step 3
    circ = Circuit()
    circ += model.cnot(
        [l[(1, 7)], l[(1, 0)], l[(0, 0)], l[(1, 1)], l[(1, 3)], l[(1, 4)]]
    )
    circ += model.idle([l[(1, 2)], l[(1, 6)], l[(1, 5)], *layout.anc_qubits])
    yield circ
    yield model.tick()

    # step 4
    circ = Circuit()
    circ += model.cnot(
        [l[(1, 0)], l[(1, 1)], l[(1, 7)], l[(0, 0)], l[(1, 3)], l[(1, 2)]]
    )
    circ += model.idle([l[(1, 6)], l[(1, 5)], l[(1, 4)], *layout.anc_qubits])
    yield circ
    yield model.tick()

    # step 5
    circ = Circuit()
    circ += model.cnot([l[(1, 7)], l[(1, 6)], l[(0, 0)], l[(1, 5)]])
    circ += model.idle(
        [l[(1, 0)], l[(1, 1)], l[(1, 2)], l[(1, 3)], l[(1, 4)], *layout.anc_qubits]
    )
    yield circ
    yield model.tick()

    # grow code to maximum size
    for d in range(3, layout.distance, 2):
        yield from grow_code_odd_d_iterator_cnots(
            model=model, layout=layout, init_distance=d
        )


def grow_code_odd_d_iterator_cnots(
    model: Model,
    layout: Layout,
    init_distance: int,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an circuit growing
    an odd-distance ``d`` rotated surface code to a ``d+2`` one of the given model
    without the detectors. Note that this circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a rotated surface code.

    Notes
    -----
    The implementation follows Figure 2 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}."
        )
    if layout.distance % 2 == 0:
        raise TypeError(
            f"The given layout does not have odd distance, but distance {layout.distance}."
        )
    if init_distance % 2 == 0:
        raise TypeError(f"'init_distance' must be odd, but {init_distance} was given.")
    if init_distance >= layout.distance:
        raise TypeError(
            f"'init_distance' ({init_distance}) must be strictly larger than "
            f"the code distance ({layout.distance})."
        )

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    l: dict[tuple[int, int], str] = {}
    for qubit in layout.data_qubits:
        glabel = layout.param(gate_label, qubit)["label"]
        if glabel is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        l[glabel] = qubit

    inner_ring = (init_distance - 1) // 2
    outer_ring = inner_ring + 1
    length = init_distance + 1
    qubits = set(layout.data_qubits)

    # step 1
    reset_x: list[str] = []
    reset_z: list[str] = []
    for i in [0, 2]:
        reset_z.append(l[(outer_ring, 0 + i * length)])
        reset_z.append(l[(outer_ring, length - 1 + i * length)])
        reset_x += [l[(outer_ring, p + i * length)] for p in range(1, length - 2, 2)]
        reset_z += [l[(outer_ring, p + i * length)] for p in range(2, length - 1, 2)]
    for i in [1, 3]:
        reset_x.append(l[(outer_ring, 0 + i * length)])
        reset_x.append(l[(outer_ring, length - 1 + i * length)])
        reset_z += [l[(outer_ring, p + i * length)] for p in range(1, length - 2, 2)]
        reset_x += [l[(outer_ring, p + i * length)] for p in range(2, length - 1, 2)]
    idling = qubits - set(reset_x) - set(reset_z)

    yield model.reset_x(reset_x) + model.reset_z(reset_z) + model.idle(idling)
    yield model.tick()

    # step 2
    cnots: list[str] = []
    for i in range(4):
        odd = [l[(outer_ring, p + i * length)] for p in range(1, length - 2, 2)]
        even = [l[(outer_ring, p + i * length)] for p in range(2, length - 1, 2)]
        if i % 2 == 1:
            odd, even = even, odd
        cnots += list(chain.from_iterable(zip(odd, even)))

        last = l[(outer_ring, length - 1 + i * length)]
        inner = l[(inner_ring, (length - 2 + i * (length - 2)) % (4 * length - 8))]
        if i % 2 == 1:
            last, inner = inner, last
        cnots += [inner, last]
    idling = qubits - set(cnots)

    yield model.cnot(cnots) + model.idle(idling)
    yield model.tick()

    # step 3
    cnots = []
    for i in range(4):
        first = l[(outer_ring, 0 + length * i)]
        last = l[(outer_ring, (length - 1 + length * ((i - 1) % 4)))]
        if i % 2 == 1:
            first, last = last, first
        cnots += [last, first]

        outer = [l[(outer_ring, p + i * length)] for p in range(1, length - 1)]
        inner = [l[(inner_ring, p + i * (length - 2))] for p in range(0, length - 2)]
        if i % 2 == 1:
            outer, inner = inner, outer
        cnots += list(chain.from_iterable(zip(inner, outer)))
    idling = qubits - set(cnots)

    yield model.cnot(cnots) + model.idle(idling)
    yield model.tick()


@qubit_encoding
def encoding_qubits_x0_iterator_cnots(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +X eigenstate to a distance-3 rotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    yield from encoding_qubits_iterator_cnots(
        model=model, layout=layout, physical_reset_op="reset_x"
    )


@qubit_encoding
def encoding_qubits_y0_iterator_cnots(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +Y eigenstate to a distance-3 rotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    yield from encoding_qubits_iterator_cnots(
        model=model, layout=layout, physical_reset_op="reset_y"
    )


@qubit_encoding
def encoding_qubits_z0_iterator_cnots(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +Z eigenstate to a distance-3 rotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    yield from encoding_qubits_iterator_cnots(
        model=model, layout=layout, physical_reset_op="reset_z"
    )


gate_to_iterator = {
    "TICK": qec_round_iterator,
    "I": idle_iterator,
    "S": log_fold_trans_s_iterator,
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
gate_to_iterator_mid_cycle_cnots = {
    "CX": log_trans_cnot_mid_cycle_css_iterator,
    "CNOT": log_trans_cnot_mid_cycle_css_iterator,
    "R": init_qubits_z0_iterator,
    "RZ": init_qubits_z0_iterator,
    "RX": init_qubits_x0_iterator,
    "M": log_meas_z_iterator,
    "MZ": log_meas_z_iterator,
    "MX": log_meas_x_iterator,
}
tick_iterators_mid_cycle_cnots = [
    to_mid_cycle_iterator_cnots,
    to_end_cycle_iterator_cnots,
]
gate_to_iterator_pipelined = {
    "TICK": qec_round_pipelined_iterator,
    "I": idle_iterator,
    "S": log_fold_trans_s_iterator,
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

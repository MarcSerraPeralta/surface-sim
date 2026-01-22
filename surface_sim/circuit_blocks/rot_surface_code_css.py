from collections.abc import Generator, Collection
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


def encoding_qubits_d3_iterator_cnots(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to a distance-3 rotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 rotated surface code.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a distance-3 rotated surface code.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}"
        )
    if layout.distance != 3:
        raise TypeError(
            f"The given layout must have distance=3, but distance={layout.distance} was given."
        )

    # map layout qubits to the Figure 1 from the reference
    # mm = middle-middle, tr = top right, mr = middle-right...
    map: dict[str, str] = {}
    map["mm"] = [q for q in layout.data_qubits if len(layout.get_neighbors([q])) == 4][
        0
    ]
    for x_anc in layout.get_qubits(role="anc", stab_type="x_type"):
        if len(layout.get_neighbors([x_anc])) != 2:
            continue
        q_corner, q_middle = layout.get_neighbors([x_anc])
        if len(layout.get_neighbors([q_corner])) != 2:
            q_corner, q_middle = q_middle, q_corner
        if "tr" in map:
            map["bl"] = q_corner
            map["bm"] = q_middle
        else:
            map["tr"] = q_corner
            map["tm"] = q_middle
    for z_anc in layout.get_qubits(role="anc", stab_type="z_type"):
        if len(layout.get_neighbors([z_anc])) != 2:
            continue
        q_corner, q_middle = layout.get_neighbors([z_anc])
        if len(layout.get_neighbors([q_corner])) != 2:
            q_corner, q_middle = q_middle, q_corner
        if "tl" in map:
            map["br"] = q_corner
            map["mr"] = q_middle
        else:
            map["tl"] = q_corner
            map["ml"] = q_middle

    # step 1
    circ = Circuit()
    circ += model.reset_z([map["tl"], map["tr"], map["bl"], map["br"]])
    circ += model.reset_x([map["tm"], map["ml"], map["mr"], map["bm"]])
    circ += model.__getattribute__(physical_reset_op)([map["mm"]])
    yield circ + model.idle(layout.anc_qubits)
    yield model.tick()

    # step 2
    circ = Circuit()
    circ += model.cnot(
        [map["tm"], map["tr"], map["mr"], map["mm"], map["bm"], map["bl"]]
    )
    circ += model.idle([map["tl"], map["ml"], map["br"], *layout.anc_qubits])
    yield circ
    yield model.tick()

    # step 3
    circ = Circuit()
    circ += model.cnot(
        [map["ml"], map["tl"], map["mm"], map["tm"], map["mr"], map["br"]]
    )
    circ += model.idle([map["tr"], map["bl"], map["bm"], *layout.anc_qubits])
    yield circ
    yield model.tick()

    # step 4
    circ = Circuit()
    circ += model.cnot(
        [map["tl"], map["tm"], map["ml"], map["mm"], map["mr"], map["tr"]]
    )
    circ += model.idle([map["bl"], map["bm"], map["br"], *layout.anc_qubits])
    yield circ
    yield model.tick()

    # step 5
    circ = Circuit()
    circ += model.cnot([map["ml"], map["bl"], map["mm"], map["bm"]])
    circ += model.idle(
        [map["tl"], map["tm"], map["tr"], map["mr"], map["br"], *layout.anc_qubits]
    )
    yield circ
    yield model.tick()


@qubit_encoding
def encoding_qubits_x0_d3_iterator_cnots(
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
    yield from encoding_qubits_d3_iterator_cnots(
        model=model, layout=layout, physical_reset_op="reset_x"
    )


@qubit_encoding
def encoding_qubits_y0_d3_iterator_cnots(
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
    yield from encoding_qubits_d3_iterator_cnots(
        model=model, layout=layout, physical_reset_op="reset_y"
    )


@qubit_encoding
def encoding_qubits_z0_d3_iterator_cnots(
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
    yield from encoding_qubits_d3_iterator_cnots(
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

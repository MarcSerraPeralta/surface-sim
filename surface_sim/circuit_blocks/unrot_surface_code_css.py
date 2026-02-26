from collections.abc import Collection, Generator
from stim import Circuit

from ..detectors import Detectors
from ..layouts.layout import Layout
from ..models import Model
from .decorators import qubit_encoding

# methods to have in this script
from .util import (
    idle_iterator,
    init_qubits,
    init_qubits_iterator,
    init_qubits_x0_iterator,
    init_qubits_x1_iterator,
    init_qubits_z0_iterator,
    init_qubits_z1_iterator,
    log_depolarize1_error_iterator,
    log_fold_trans_h,
    log_fold_trans_h_iterator,
    log_fold_trans_s,
    log_fold_trans_s_iterator,
    log_meas,
    log_meas_iterator,
    log_meas_x_iterator,
    log_meas_z_iterator,
    log_trans_cnot,
    log_trans_cnot_iterator,
    log_x,
    log_x_error_iterator,
    log_x_iterator,
    log_y_error_iterator,
    log_z,
    log_z_error_iterator,
    log_z_iterator,
    qec_round_iterator,
    qec_round_iterator_cnots,
    qubit_coords,
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
    "qec_round_cnots",
    "qec_round_iterator_cnots",
    "encoding_qubits_iterator",
    "encoding_qubits_x0_iterator",
    "encoding_qubits_y0_iterator",
    "encoding_qubits_z0_iterator",
    "encoding_qubits_iterator_cnots",
    "encoding_qubits_x0_iterator_cnots",
    "encoding_qubits_y0_iterator_cnots",
    "encoding_qubits_z0_iterator_cnots",
    "log_x_error_iterator",
    "log_y_error_iterator",
    "log_z_error_iterator",
    "log_depolarize1_error_iterator",
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


def qec_round_cnots(
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
    This implementation uses the following instructions: CNOT, RZ, RX, MZ, MX.
    Note that if ``anc_reset = False``, then the ancillas are not reset in the first round
    and stim assumes that, if not specified, they are reset in the Z-basis, which is the
    incorrect basis for the X-type ancillas. See the initialization iterators from the
    dodecahedron code.

    It activates all the ancillas in ``detectors`` to always build the detectors.
    As this function should not be used when building encoded circuits with
    the iterating functions, it does not matter if the detectors are activated or not.
    """
    circuit = sum(
        qec_round_iterator_cnots(model=model, layout=layout, anc_reset=anc_reset),
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


def _encoding_qubits_iterator(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
    primitive_gates: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to an unrotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a unrotated surface code.
    primitive_gates
        Set of primitive gates to use. The available options are:
        (1) ``"cnot"``, which uses Rx, RZ, and CNOT gates, and
        (2) ``"cz"``, which uses RZ, H, and CZ gates.
        Note that the ``physical_reset_op`` will not be decomposed into primitive
        gates.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    circ = model.reset(layout.qubits)
    centre_qubit = _map_coordinates_to_qubits(layout, [[(0,0)]])[0]
    circ += model.__getattribute__(physical_reset_op)(centre_qubit)
    yield circ
    yield model.tick()

    if layout.distance % 2 == 1:
        encoding_distance_3_gates = _map_coordinates_to_qubits(layout, encoding_distance_3_gates_coord)
        yield from _circuit_builder_iterator(encoding_distance_3_gates, model, layout, primitive_gates)
    else:
        encoding_distance_2_gates = _map_coordinates_to_qubits(layout, encoding_distance_2_gates_coord)
        yield from _circuit_builder_iterator(encoding_distance_2_gates, model, layout, primitive_gates)
        if layout.distance >= 4:
            encoding_distance_4_gates = _map_coordinates_to_qubits(layout, encoding_distance_4_gates_coord)
            yield from _circuit_builder_iterator(encoding_distance_4_gates, model, layout, primitive_gates)
    for d in range(4-layout.distance%2, layout.distance, 2):
        gates_list_coords = _grow_code_coordinates(layout, d, primitive_gates)
        gates_list = _map_coordinates_to_qubits(layout, gates_list_coords)
        yield from _circuit_builder_iterator(gates_list, model, layout, primitive_gates)

def _circuit_builder_iterator(gates_list: list[list[str]], model: Model, layout: Layout, primitive_gates: str):
    """
    Take the primative gate and a list of lists of two qubit gates to be applied at each step 
    and yields the corresponding iterator for the circuit. 
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    gates_list
        List of lists of gates to be applied at each step. 
        The first list corresponds to the Hadamard gates, the rest corresponds to two qubit gates.
    model
        Noise model for the gates.
    layout
        Code layout.
    primitive_gates
        Set of primitive gates to use. The available options are:
        (1) ``"cnot"``, which uses Rx, RZ, and CNOT gates, and
        (2) ``"cz"``, which uses RZ, H, and CZ gates.
        Note that the ``physical_reset_op`` will not be decomposed into primitive
        gates.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    if primitive_gates not in ["cnot", "cz"]:
        raise ValueError(f"'{primitive_gates}' is not available as primitive gate set.")
    qubits = layout.qubits
    h_gates = gates_list[0]
    tq_gates_list = gates_list[1:]
    # if the primitive gate is cz, then the h gates will be applied to the target qubits of the cz gates, 
    # which are the second qubits in the two qubit gates.
    # we take symmetric difference between the target qubits of the current two qubit gates and the next two qubit gates
    if primitive_gates == "cz":
        h_0 = tq_gates_list[0][1::2]
        yield model.h_gate(h_gates+h_0) + model.idle(set(qubits) - set(h_gates+h_0))
    else:
        yield model.h_gate(h_gates) + model.idle(set(qubits) - set(h_gates))
    yield model.tick()
    for i in range(len(tq_gates_list)):
        if primitive_gates == "cz":
            yield model.cz(tq_gates_list[i]) + model.idle(set(qubits) - set(tq_gates_list[i]))
        else:
            yield model.cnot(tq_gates_list[i]) + model.idle(set(qubits) - set(tq_gates_list[i]))
        yield model.tick()
        if primitive_gates == "cz":
            h_1 = set(tq_gates_list[i][1::2])
            if i != len(tq_gates_list) - 1:
                h_1 = h_1^set(tq_gates_list[i+1][1::2])
            yield model.h_gate(h_1) + model.idle(set(qubits) - set(h_1))
            yield model.tick()
        
def _map_coordinates_to_qubits(layout: Layout, gates_coord_list: list[list[tuple[int, int]]]) -> list[list[str]]:
    """
    Convert a list of lists of coordinates to a list of lists of qubits corresponding to the coordinates
    that was defined in the log_gates when setting up the layout.

    Parameters
    ----------
    layout
        Code layout.
    gates_coord_list
        List of lists of coordinates corresponding to the gates to be applied at each step.

    Notes
    -----
    """
    l: dict[tuple[int, int], str] = {}
    for qubit in layout.data_qubits:
        glabel = layout.param(f"encoding_{layout.logical_qubits[0]}", qubit)["label"]
        if glabel is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"encoding_{layout.logical_qubits[0]} gate on qubit {qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        l[glabel] = qubit
    
    gates_list = []
    for gates_coord in gates_coord_list:
        gates_list.append([l[coord] for coord in gates_coord])
    return gates_list

def _grow_code_coordinates(
    layout: Layout,
    cur_d: int,
    primitive_gates: str,
) -> list[list[tuple[int, int]]]:
    """
    Yields list of lists of gates in the coordinate form which as a whole correspond to a circuit that grows
    a distance ``d`` unrotated surface code to a ``d+2`` one of the given model
    without the detectors. Note that this circuit is not fault tolerant.

    Parameters
    ----------
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to an unrotated surface code.
    primitive_gates
        Set of primitive gates to use. The available options are:
        (1) ``"cnot"``, which uses Rx, RZ, and CNOT gates, and
        (2) ``"cz"``, which uses RZ, H, and CZ gates.
        Note that the ``physical_reset_op`` will not be decomposed into primitive
        gates.

    Notes
    -----
    The implementation follows Figure 2 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    if layout.code != "unrotated_surface_code":
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}."
        )
    if cur_d % 2 != layout.distance % 2:
        raise TypeError(
            f"'current distance' and 'layout_distance' must have the same parity, but current distance:{cur_d}, layout.distance:{layout.distance} were given."
        )
    if cur_d >= layout.distance:
        raise TypeError(
            f"'current distance' ({cur_d}) must be strictly smaller than "
            f"the code distance ({layout.distance})."
        )
    if primitive_gates not in ["cnot", "cz"]:
        raise ValueError(f"'{primitive_gates}' is not available as primitive gate set.")

    gates_list = []
    h_gates_coord = []
    for i in range(1-cur_d, cur_d +1, 2):
        h_gates_coord += [(i, -cur_d-1)]  # top
        h_gates_coord += [(i, cur_d+1)]  # bottom
    for i in range(-cur_d, cur_d+2, 2):
        h_gates_coord += [(-cur_d, i)]  # left
        h_gates_coord += [(cur_d, i)]  # right
    gates_list.append(h_gates_coord)
    # step 1, two qubit gates at 4 corner
    tq_gates_1_coord = [
            (-cur_d, -cur_d),
            (-cur_d-1, -cur_d-1),
            (cur_d, -cur_d),
            (cur_d+1, -cur_d-1),
            (-cur_d, cur_d),
            (-cur_d-1, cur_d+1),
            (cur_d, cur_d),
            (cur_d+1,cur_d+1)]
    # two qubit gates at outer line
    for i in range(1-cur_d, cur_d+1, 2):
        tq_gates_1_coord += [(1-cur_d, i), (-cur_d-1, i)]  # left
        tq_gates_1_coord += [(cur_d-1, i),(cur_d+1, i)]  # right
    # two qubit gates at inner corner
    for i in range(2-cur_d, cur_d, 2):
        tq_gates_1_coord += [(2-cur_d, i),(-cur_d, i)]  # left
        tq_gates_1_coord += [(cur_d-2, i),(cur_d, i)]  # right
    gates_list.append(tq_gates_1_coord)
    # step 2, two qubit gates at 4 corner
    tq_gates_2_coord = [
            (-cur_d, -cur_d),
            (-cur_d-1, -cur_d+1),
            (cur_d, -cur_d),
            (cur_d+1, 1-cur_d),
            (-cur_d, cur_d),
            (-cur_d-1, cur_d-1),
            (cur_d, cur_d),
            (cur_d+1, cur_d-1),
        ]
    # two qubit gates at outer line
    for i in range(1-cur_d, cur_d+1, 2):
        tq_gates_2_coord += [(i, -cur_d-1),(i, 1-cur_d)]  # top
        tq_gates_2_coord += [(i, cur_d+1),(i, cur_d-1)]  # bottom
    # two qubit gates at inner corner
    for i in range(2-cur_d, cur_d, 2):
        tq_gates_2_coord += [(i, -cur_d),(i, 2-cur_d)]  # top
        tq_gates_2_coord += [(i, cur_d),(i, cur_d-2)]  # bottom
    gates_list.append(tq_gates_2_coord)
    # step 3
    tq_gates_3_coord = []
    for i in range(1-cur_d, cur_d+1, 2):
        tq_gates_3_coord += [(i, -cur_d-1), (i + 1, -cur_d)]  # top
        tq_gates_3_coord += [(i, cur_d+1),(i - 1, cur_d)]  # bottom
    for i in range(2-cur_d, cur_d, 2):
        tq_gates_3_coord += [(-cur_d, i),(-cur_d-1, i - 1)]  # left
        tq_gates_3_coord += [(cur_d, i),(cur_d+1, i + 1)]  # right
    gates_list.append(tq_gates_3_coord)
    # step 4
    tq_gates_4_coord = []
    for i in range(1-cur_d, cur_d+1, 2):
        tq_gates_4_coord += [(i, -cur_d-1),(i - 1, -cur_d)]  # top
        tq_gates_4_coord += [(i, cur_d+1),(i + 1, cur_d)]  # bottom
    for i in range(2-cur_d, cur_d, 2):
        tq_gates_4_coord += [(-cur_d, i),(-cur_d-1, i + 1)]  # left
        tq_gates_4_coord += [(cur_d, i),(cur_d+1, i - 1)]  # right
    gates_list.append(tq_gates_4_coord)

    return gates_list

def encoding_qubits_iterator(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to an unrotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to an unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    but uses RZ, H, and CZ operations as primitive operations, except for the
    ``physical_reset_op``.
    """
    yield from _encoding_qubits_iterator(
        model=model,
        layout=layout,
        physical_reset_op=physical_reset_op,
        primitive_gates="cz",
    )


@qubit_encoding
def encoding_qubits_x0_iterator(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +X eigenstate to an unrotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 unrotated surface code.

    Notes
    -----
    The implementation follows Figure 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    but uses RZ, H, and CZ operations as primitive operations, except for the
    ``physical_reset_op``.
    """
    yield from encoding_qubits_iterator(
        model=model, layout=layout, physical_reset_op="reset_x"
    )


@qubit_encoding
def encoding_qubits_y0_iterator(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +Y eigenstate to an unrotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    but uses RZ, H, and CZ operations as primitive operations, except for the
    ``physical_reset_op``.
    """
    yield from encoding_qubits_iterator(
        model=model,
        layout=layout,
        physical_reset_op="reset_y",
    )


@qubit_encoding
def encoding_qubits_z0_iterator(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +Z eigenstate to an unrotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    but uses RZ, H, and CZ operations as primitive operations, except for the
    ``physical_reset_op``.
    """
    yield from encoding_qubits_iterator(
        model=model,
        layout=layout,
        physical_reset_op="reset_z",
    )


def encoding_qubits_iterator_cnots(
    model: Model,
    layout: Layout,
    physical_reset_op: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to an unrotated surface code of the given model without the detectors.
    Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to an unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    yield from _encoding_qubits_iterator(
        model=model,
        layout=layout,
        physical_reset_op=physical_reset_op,
        primitive_gates="cnot",
    )


@qubit_encoding
def encoding_qubits_x0_iterator_cnots(
    model: Model,
    layout: Layout,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    for the +X eigenstate to an unrotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

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
    for the +Y eigenstate to an unrotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

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
    for the +Z eigenstate to an unrotated surface code of the given model
    without the detectors. Note that this encoding circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout. Must correspond to a distance-3 unrotated surface code.

    Notes
    -----
    The implementation follows Figure 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    yield from encoding_qubits_iterator_cnots(
        model=model, layout=layout, physical_reset_op="reset_z"
    )


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
    "X_ERROR": log_x_error_iterator,
    "Y_ERROR": log_y_error_iterator,
    "Z_ERROR": log_z_error_iterator,
    "DEPOLARIZE1": log_depolarize1_error_iterator,
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
    "X_ERROR": log_x_error_iterator,
    "Y_ERROR": log_y_error_iterator,
    "Z_ERROR": log_z_error_iterator,
    "DEPOLARIZE1": log_depolarize1_error_iterator,
}
encoding_distance_2_gates_coord = [
    [(1,-1), (-1,1)],
    [(0,0),(-1,-1),(1,-1),(1,1)],
    [(-1,1),(-1,-1),(0,0),(1,1)],
    [(-1,1), (0,0)],
    [(1,-1), (0,0)],
    ]
encoding_distance_3_gates_coord = [
    [(0,-2),(1,1),(1,-1),(-1,1),(-1,-1),(0,2)],
    [(-1,-1),(-2,-2),(1,-1),(2,-2),(0,0),(-2,0),(-1,1),(-2,2),(1,1),(2,2)],
    [(0,0),(2,0),(-1,-1),(-2,0)],
    [(0,-2),(0,0),(-1,1),(-2,0),(1,-1),(2,0)],
    [(0,2),(0,0),(1,1),(2,0),(0,-2),(-1,-1)],
    [(0,-2),(1,-1),(0,2),(-1, 1)],
    [(0,2),(1,1)],
    ]
encoding_distance_4_gates_coord = [
    [(-1,-3),(1,-3),(-2,-2),(2,-2),(-2,0),(2,0),(-2,2),(2,2),(-1,3),(1,3)],
    [(-2,-2),(-3,-3),(2,-2),(3,-3),(-1,-1),(-3,-1),(0,-2),(1,-1),(0,0),(-2,0),(0,2),(-1,1),(1,1),(3,1),(-2,2),(-3,3),(2,2),(3,3)],
    [(-2,-2),(-3,-1),(-1,-3),(-1,-1),(1,-3),(0,-2),(1,-1),(3,-1),(0,0),(2,0),(-1,1),(-3,1),(-1,3),(0,2),(1,3),(1,1),(2,2),(3,1)],
    [(-1,-3),(0,-2),(1,-3),(1,-1),(2,-2),(3,-1),(-2,0),(-3,-1),(2,0),(3,1),(-2,2),(-3,1),(-1,3),(-1,1),(1,3),(0,2)],
    [(-1,-3),(-2,-2),(1,-3),(2,-2),(-2,0),(-3,1),(2,0),(3,-1),(-1,3),(-2,2),(1,3),(2,2)],
]

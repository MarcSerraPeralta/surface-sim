from collections.abc import Collection, Generator, Sequence

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
        (1) ``"cnot"``, which uses RZ, RX, and CNOT gates, and
        (2) ``"cz"``, which uses RZ, H, and CZ gates.
        Note that the ``physical_reset_op`` will not be decomposed into primitive
        gates.

    Notes
    -----
    The implementation follows Figures 2 and 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    if layout.code != "unrotated_surface_code":
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}."
        )
    if primitive_gates not in ["cnot", "cz"]:
        raise ValueError(f"'{primitive_gates}' is not available as primitive gate set.")

    if layout.distance % 2 == 1:
        reset_x = [(0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2)]
        reset_z = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        cnot_gates = [
            [(-1, -1), (-2, -2), (1, -1), (2, -2), (0, 0), (-2, 0), (-1, 1), (-2, 2), (1, 1), (2, 2)], 
            [(0, 0), (2, 0), (-1, -1), (-2, 0)],
            [(0, -2), (0, 0), (-1, 1), (-2, 0), (1, -1), (2, 0)],
            [(0, 2), (0, 0), (1, 1), (2, 0), (0, -2), (-1, -1)],
            [(0, -2), (1, -1), (0, 2), (-1, 1)],
            [(0, 2), (1, 1)],
        ]  # fmt: skip
        yield from _grow_code_iterator(
            model, layout, reset_x, reset_z, cnot_gates, physical_reset_op, primitive_gates
        )  # fmt: skip
    else:
        reset_x = [(1, -1), (-1, 1)]
        reset_z = [(1, 1), (-1, -1)]
        cnot_gates = [
            [(0, 0), (-1, -1), (1, -1), (1, 1)],
            [(-1, 1), (-1, -1), (0, 0), (1, 1)],
            [(-1, 1), (0, 0)],
            [(1, -1), (0, 0)],
        ]
        yield from _grow_code_iterator(
            model, layout, reset_x, reset_z, cnot_gates, physical_reset_op, primitive_gates
        )  # fmt: skip

        # growing circuit from d=2 to 4 is different than the rest, see Fig 9(c).
        if layout.distance >= 4:
            reset_x = [(-1, -3), (1, -3), (-2, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (2, 2), (-1, 3), (1, 3)]  # fmt: skip
            reset_z = [(3, 3), (-3, 3), (3, -3), (-3, -3), (3, 1), (-3, 1), (3, -1), (-3, -1), (0, 2), (0, -2)]  # fmt: skip
            cnot_gates = [
                [(-2, -2), (-3, -3), (2, -2), (3, -3), (-1, -1), (-3, -1), (0, -2), (1, -1), (0, 0), (-2, 0), (0, 2), (-1, 1), (1, 1), (3, 1), (-2, 2), (-3, 3), (2, 2), (3, 3)],
                [(-2, -2), (-3, -1), (-1, -3), (-1, -1), (1, -3), (0, -2), (1, -1), (3, -1), (0, 0), (2, 0), (-1, 1), (-3, 1), (-1, 3), (0, 2), (1, 3), (1, 1), (2, 2), (3, 1)],
                [(-1, -3), (0, -2), (1, -3), (1, -1), (2, -2), (3, -1), (-2, 0), (-3, -1), (2, 0), (3, 1), (-2, 2), (-3, 1), (-1, 3), (-1, 1), (1, 3), (0, 2)],
                [(-1, -3), (-2, -2), (1, -3), (2, -2), (-2, 0), (-3, 1), (2, 0), (3, -1), (-1, 3), (-2, 2), (1, 3), (2, 2)],
            ]  # fmt: skip
            yield from _grow_code_iterator(
                model, layout, reset_x, reset_z, cnot_gates, None, primitive_gates
            )

    for d in range(4 - layout.distance % 2, layout.distance, 2):
        reset_x, reset_z, cnot_gates = _grow_code_coordinates(d)
        yield from _grow_code_iterator(
            model, layout, reset_x, reset_z, cnot_gates, None, primitive_gates
        )


def _grow_code_iterator(
    model: Model,
    layout: Layout,
    reset_x_coords: Collection[tuple[int, int]],
    reset_z_coords: Collection[tuple[int, int]],
    cnot_layers_coords: Sequence[Sequence[tuple[int, int]]],
    physical_reset_op: str | None,
    primitive_gates: str,
):
    """
    Yields stim blocks corresponding to the growth of the code following
    the given RX, RZ, and CNOT operatons.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    reset_x_coords
        Generalized coordinates for the data qubits that need to be reset in the X basis.
    reset_z_coords
        Generalized coordinates for the data qubits that need to be reset in the Z basis.
    cnot_layers_coords
        List of CNOT layers that need to be applied at each step of the encoding
        circuit. The data qubits are specified by the generalized coordinates.
    physical_reset_op
        Reset operation to be applied to the physical qubit that will grow
        to a unrotated surface code.
        If ``None``, this circuit grows an already existing code to one of higher distance.
    primitive_gates
        Set of primitive gates to use. The available options are:
        (1) ``"cnot"``, which uses RZ, RX, CNOT gates, and
        (2) ``"cz"``, which uses RZ, H, and CZ gates.

    Notes
    -----
    The implementation follows Figure 2 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    gate_label = f"encoding_{layout.logical_qubits[0]}"

    l: dict[tuple[int, int], str] = {}
    for data_qubit in layout.data_qubits:
        glabel = layout.param(gate_label, data_qubit)["label"]
        if glabel is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"an encoding circuit on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        l[glabel] = data_qubit

    qubits = set(layout.qubits)

    # step 1: resets
    circ = Circuit()
    reset_x = [l[c] for c in reset_x_coords]
    reset_z = [l[c] for c in reset_z_coords]
    hadamards_prev = hadamards_curr = set(reset_x)
    if primitive_gates == "cnot":
        circ += model.reset_x(reset_x) + model.reset_z(reset_z)
    elif primitive_gates == "cz":
        circ += model.reset_z(reset_z + reset_x)
    exec_qubits = reset_x + reset_z
    if physical_reset_op is not None:
        circ += model.__getattribute__(physical_reset_op)([l[(0, 0)]])
        exec_qubits.append(l[(0, 0)])
    yield circ + model.idle(qubits - set(exec_qubits))
    yield model.tick()

    # steps 2, 3, ...: cnot gates
    for cnot_pairs_coords in cnot_layers_coords:
        cnot_pairs = [l[c] for c in cnot_pairs_coords]
        hadamards_curr = set(cnot_pairs[1::2])

        # apply hadamards between step prev and curr
        if primitive_gates == "cz":
            hadamards = hadamards_prev.symmetric_difference(hadamards_curr)
            yield model.hadamard(hadamards) + model.idle(qubits - hadamards)
            yield model.tick()
        hadamards_prev = hadamards_curr

        circ = Circuit()
        if primitive_gates == "cnot":
            circ += model.cnot(cnot_pairs)
        elif primitive_gates == "cz":
            circ += model.cphase(cnot_pairs)
        circ += model.idle(qubits - set(cnot_pairs))
        yield circ
        yield model.tick()

    if primitive_gates == "cz":
        yield model.hadamard(hadamards_curr) + model.idle(qubits - hadamards_curr)
        yield model.tick()


def _grow_code_coordinates(
    d: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """
    Returns the reset and CNOT information to grow a distance ``d > 4`` unrotated
    surface code to a ``d+2`` one. Note that this circuit is not fault tolerant.

    Parameters
    ----------
    d
        Distance of the current unrotated surface code.

    Returns
    -------
    reset_x_coords
        Generalized coordinates for the data qubits that need to be reset in the X basis.
    reset_z_coords
        Generalized coordinates for the data qubits that need to be reset in the Z basis.
    cnot_layers_coords
        List of CNOT layers that need to be applied at each step of the encoding
        circuit. The data qubits are specified by the generalized coordinates.

    Notes
    -----
    The implementation follows Figure 2 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    reset_x_coords = []
    for i in range(-d + 1, d + 1, 2):
        reset_x_coords += [(i, -d - 1)]  # top
        reset_x_coords += [(i, d + 1)]  # bottom
    for i in range(-d, d + 2, 2):
        reset_x_coords += [(-d, i)]  # left
        reset_x_coords += [(d, i)]  # right

    reset_z_coords = []
    for i in range(-d + 2, d, 2):
        reset_z_coords += [(i, -d)]  # top
        reset_z_coords += [(i, d)]  # bottom
    for i in range(-d - 1, d + 3, 2):
        reset_z_coords += [(-d - 1, i)]  # left
        reset_z_coords += [(d + 1, i)]  # right

    cnot_layers_coords = []

    # step 1
    step1 = [(-d, -d), (-d - 1, -d - 1), (d, -d), (d + 1, -d - 1), (-d, d), (-d - 1, d + 1), (d, d), (d + 1, d + 1)]  # fmt: skip
    for i in range(1 - d, d + 1, 2):
        step1 += [(1 - d, i), (-d - 1, i)]  # left
        step1 += [(d - 1, i), (d + 1, i)]  # right
    for i in range(2 - d, d, 2):
        step1 += [(2 - d, i), (-d, i)]  # left
        step1 += [(d - 2, i), (d, i)]  # right
    cnot_layers_coords.append(step1)

    # step 2
    step2 = [(-d, -d), (-d - 1, -d + 1), (d, -d), (d + 1, 1 - d), (-d, d), (-d - 1, d - 1), (d, d), (d + 1, d - 1)]  # fmt: skip
    for i in range(1 - d, d + 1, 2):
        step2 += [(i, -d - 1), (i, 1 - d)]  # top
        step2 += [(i, d + 1), (i, d - 1)]  # bottom
    for i in range(2 - d, d, 2):
        step2 += [(i, -d), (i, 2 - d)]  # top
        step2 += [(i, d), (i, d - 2)]  # bottom
    cnot_layers_coords.append(step2)

    # step 3
    step3 = []
    for i in range(1 - d, d + 1, 2):
        step3 += [(i, -d - 1), (i + 1, -d)]  # top
        step3 += [(i, d + 1), (i - 1, d)]  # bottom
    for i in range(2 - d, d, 2):
        step3 += [(-d, i), (-d - 1, i - 1)]  # left
        step3 += [(d, i), (d + 1, i + 1)]  # right
    cnot_layers_coords.append(step3)

    # step 4
    step4 = []
    for i in range(1 - d, d + 1, 2):
        step4 += [(i, -d - 1), (i - 1, -d)]  # top
        step4 += [(i, d + 1), (i + 1, d)]  # bottom
    for i in range(2 - d, d, 2):
        step4 += [(-d, i), (-d - 1, i + 1)]  # left
        step4 += [(d, i), (d + 1, i - 1)]  # right
    cnot_layers_coords.append(step4)

    return reset_x_coords, reset_z_coords, cnot_layers_coords


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
    The implementation follows Figures 2 and 9 from:

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
        Code layout.

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
        Code layout.

    Notes
    -----
    The implementation follows Figures 2 and 9 from:

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
        Code layout.

    Notes
    -----
    The implementation follows Figures 2 and 9 from:

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
    The implementation follows Figures 2 and 9 from:

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
        Code layout.

    Notes
    -----
    The implementation follows Figures 2 and 9 from:

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
        Code layout.

    Notes
    -----
    The implementation follows Figures 2 and 9 from:

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
        Code layout.

    Notes
    -----
    The implementation follows Figures 2 and 9 from:

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

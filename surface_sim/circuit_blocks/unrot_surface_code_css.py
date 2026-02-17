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
    The implementation follows Figure 2, 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    if layout.distance % 2 == 1:
        yield from _encoding_qubits_odd_d_iterator(
            model=model,
            layout=layout,
            physical_reset_op=physical_reset_op,
            primitive_gates=primitive_gates,
        )
    else:
        yield from _encoding_qubits_even_d_iterator(
            model=model,
            layout=layout,
            physical_reset_op=physical_reset_op,
            primitive_gates=primitive_gates,
        )

def _encoding_qubits_odd_d_iterator(
    model: Model, layout: Layout, physical_reset_op: str, primitive_gates: str
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to a odd-distance unrotated surface code of the given model without the detectors.
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
    The implementation follows Figure 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    if layout.code != "unrotated_surface_code":
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}."
        )
    if layout.distance % 2 == 0:
        raise TypeError(
            f"The given layout does not have odd distance, but distance {layout.distance}."
        )
    if primitive_gates not in ["cnot", "cz"]:
        raise ValueError(f"'{primitive_gates}' is not available as primitive gate set.")
    if len(layout.qubit_coords.items()) == 0:
        raise ValueError("The layout does not have qubit coordinates.")
        
    gate_label = f"encoding_{layout.logical_qubits[0]}"

    # maps from coordinates to qubit labels
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
    # l: dict[tuple[int, int], str] = {}
    # for qubit, coord in layout.qubit_coords.items():
    #     l[(int(coord[0]), int(coord[1]))] = qubit

    qubits = set(layout.qubits)
    d = layout.distance
    offset = d - 3 

    #encode initial d=3 code on the center of the layout 
    if primitive_gates == "cz":
        yield model.reset(qubits)
        yield model.tick()
        # H gate
        h_gates = [l[coord] for coord in [(2 + offset,0 + offset), (1 + offset,1 + offset),
        (3 + offset,1 + offset), (1 + offset,3 + offset),
        (3 + offset,3 + offset), (2 + offset,4 + offset)]]
        if physical_reset_op == "reset_x" or physical_reset_op == "reset_y":
            h_gates += [l[(2 + offset, 2 + offset)]]
        # decompose CNOT into H CZ H
        h_0 = [l[coord] for coord in [(0 + offset,0 + offset), (0 + offset,2 + offset), 
        (0 + offset,4 + offset), (4 + offset,0 + offset), 
        (4 + offset,4 + offset)]]
        if physical_reset_op == "reset_y":
            circ = model.h_gate(h_gates + h_0) + model.idle(set(qubits)-set(h_gates + h_0))
            circ += model.s_gate([l[(2 + offset, 2 + offset)]])
            yield circ
        else:
            yield model.h_gate(h_gates + h_0) + model.idle(set(qubits)-set(h_gates + h_0))
        yield model.tick()
        # Step 1
        czs_1 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,0 + offset)
        ,(3 + offset,1 + offset),(4 + offset,0 + offset) 
        ,(2 + offset,2 + offset),(0 + offset,2 + offset)
        ,(1 + offset,3 + offset),(0 + offset,4 + offset)
        ,(3 + offset,3 + offset),(4 + offset,4 + offset)]]
        h_1 = [l[coord] for coord in [(0 + offset,0 + offset), (4 + offset,2 + offset), 
        (0 + offset,4 + offset), (4 + offset,0 + offset), 
        (4 + offset,4 + offset)]]

        yield model.cz(czs_1) + model.idle(set(qubits)-set(czs_1))
        yield model.tick()
        yield model.h_gate(h_1) + model.idle(set(qubits)-set(h_1))
        yield model.tick()
        # Step 2
        czs_2 = [l[coord] for coord in [(2 + offset,2 + offset),(4 + offset,2 + offset)
        ,(1 + offset,1 + offset),(0 + offset,2 + offset)]]
        h_2 = [l[coord] for coord in [(2 + offset, 2 + offset)]]
        yield model.cz(czs_2) + model.idle(set(qubits)-set(czs_2))
        yield model.tick()
        yield model.h_gate(h_2) + model.idle(set(qubits)-set(h_2))
        yield model.tick()
        # Step 3
        czs_3 = [l[coord] for coord in [(2 + offset,0 + offset),(2 + offset,2 + offset)
        ,(1 + offset,3 + offset),(0 + offset,2 + offset)
        ,(3 + offset,1 + offset),(4 + offset,2 + offset)]]
        h_3 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,2 + offset)]]
        yield model.cz(czs_3) + model.idle(set(qubits)-set(czs_3))
        yield model.tick()
        yield model.h_gate(h_3) + model.idle(set(qubits)-set(h_3))
        yield model.tick()
        # Step 4
        czs_4 = [l[coord] for coord in [(2 + offset,4 + offset),(2 + offset,2 + offset)
        ,(3 + offset,3 + offset),(4 + offset,2 + offset)
        ,(2 + offset,0 + offset),(1 + offset,1 + offset)]]
        h_4 = [l[coord] for coord in [(1 + offset,1 + offset), (3 + offset,1 + offset), 
        (2 + offset,2 + offset), (4 + offset,2 + offset), 
        (1 + offset,3 + offset)]]
        yield model.cz(czs_4) + model.idle(set(qubits)-set(czs_4))
        yield model.tick()
        yield model.h_gate(h_4) + model.idle(set(qubits)-set(h_4))
        yield model.tick()
        # Step 5
        czs_5 = [l[coord] for coord in [(2 + offset,0 + offset),(3 + offset,1 + offset)
        ,(2 + offset,4 + offset),(1 + offset,3 + offset)]]
        h_5 = [l[coord] for coord in [(1 + offset,3 + offset),(3 + offset,1 + offset)
        ,(3 + offset,3 + offset)]]
        yield model.cz(czs_5) + model.idle(set(qubits)-set(czs_5))
        yield model.tick()
        yield model.h_gate(h_5) + model.idle(set(qubits)-set(h_5))
        yield model.tick()
        # Step 6
        czs_6 = [l[coord] for coord in [(2 + offset,4 + offset),(3 + offset,3 + offset)]]
        h_6 = [l[coord] for coord in [(3 + offset,3 + offset)]]
        yield model.cz(czs_6) + model.idle(set(qubits)-set(czs_6))
        yield model.tick()
        yield model.h_gate(h_6) + model.idle(set(qubits)-set(h_6))
        yield model.tick()
    else:
        init_x = [l[coord] for coord in [(2 + offset,0 + offset), (1 + offset,1 + offset),
        (3 + offset,1 + offset), (1 + offset,3 + offset),
        (3 + offset,3 + offset), (2 + offset,4 + offset)]]
        circ = model.reset_x(init_x) + model.reset(set(qubits)-set(init_x))
        if physical_reset_op == "reset_x":
            circ += model.reset_x([l[(2 + offset, 2 + offset)]])
        elif physical_reset_op == "reset_y":
            circ += model.reset_y([l[(2 + offset, 2 + offset)]])
        yield circ
        yield model.tick()
        # Step 1
        cnots_1 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,0 + offset)
        ,(3 + offset,1 + offset),(4 + offset,0 + offset) 
        ,(2 + offset,2 + offset),(0 + offset,2 + offset)
        ,(1 + offset,3 + offset),(0 + offset,4 + offset)
        ,(3 + offset,3 + offset),(4 + offset,4 + offset)]]
        yield model.cnot(cnots_1) + model.idle(set(qubits)-set(cnots_1))
        yield model.tick()
        # Step 2
        cnots_2 = [l[coord] for coord in [(2 + offset,2 + offset),(4 + offset,2 + offset)
        ,(1 + offset,1 + offset),(0 + offset,2 + offset)]]
        yield model.cnot(cnots_2) + model.idle(set(qubits)-set(cnots_2))
        yield model.tick()
        # Step 3
        cnots_3 = [l[coord] for coord in [(2 + offset,0 + offset),(2 + offset,2 + offset)
        ,(1 + offset,3 + offset),(0 + offset,2 + offset)
        ,(3 + offset,1 + offset),(4 + offset,2 + offset)]]
        yield model.cnot(cnots_3) + model.idle(set(qubits)-set(cnots_3))
        yield model.tick()
        # Step 4
        cnots_4 = [l[coord] for coord in [(2 + offset,4 + offset),(2 + offset,2 + offset)
        ,(3 + offset,3 + offset),(4 + offset,2 + offset)
        ,(2 + offset,0 + offset),(1 + offset,1 + offset)]]
        yield model.cnot(cnots_4) + model.idle(set(qubits)-set(cnots_4))
        yield model.tick()
        # Step 5
        cnots_5 = [l[coord] for coord in [(2 + offset,0 + offset),(3 + offset,1 + offset)
        ,(2 + offset,4 + offset),(1 + offset,3 + offset)]]
        yield model.cnot(cnots_5) + model.idle(set(qubits)-set(cnots_5))
        yield model.tick()
        # Step 6
        cnots_6 = [l[coord] for coord in [(2 + offset,4 + offset),(3 + offset,3 + offset)]]
        yield model.cnot(cnots_6) + model.idle(set(qubits)-set(cnots_6))
        yield model.tick()


    # grow code to maximum size
    for d in range(3, layout.distance, 2):
        yield from _grow_code_iterator(
            model=model, layout=layout, curr_distance=d, primitive_gates=primitive_gates
        )

def _encoding_qubits_even_d_iterator(
    model: Model, layout: Layout, physical_reset_op: str, primitive_gates: str
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an encoding circuit
    to an even-distance unrotated surface code of the given model without the detectors.
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
    primitive_gates
        Set of primitive gates to use. The available options are:
        (1) ``"cnot"``, which uses Rx, RZ, and CNOT gates, and
        (2) ``"cz"``, which uses RZ, H, and CZ gates.
        Note that the ``physical_reset_op`` will not be decomposed into primitive
        gates.

    Notes
    -----
    The implementation follows Figure 9 from:

        Higgott, Oscar. "Optimal local unitary encoding circuits for the surface code."
        Quantum 5, 517 (2021).

    """
    if layout.code != "unrotated_surface_code":
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}."
        )
    if layout.distance % 2 == 1:
        raise TypeError(
            f"The given layout does not have odd distance, but distance {layout.distance}."
        )
    if primitive_gates not in ["cnot", "cz"]:
        raise ValueError(f"'{primitive_gates}' is not available as primitive gate set.")

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    # maps from coordinates to qubit labels
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
    # l: dict[tuple[int, int], str] = {}
    # for qubit, coord in layout.qubit_coords.items():
    #     l[(int(coord[0]), int(coord[1]))] = qubit

    qubits = set(layout.qubits)
    d = layout.distance
    offset = d-2

    if primitive_gates == "cz":
        yield model.reset(qubits)
        yield model.tick()
        # H gate
        h_gates = [l[coord] for coord in [(2 + offset,0 + offset), (0 + offset,2 + offset)]]  
        if physical_reset_op == "reset_x" or physical_reset_op == "reset_y":
            h_gates += [l[(1 + offset, 1 + offset)]]      
        # decompose CNOT into H CZ H
        h_0 = [l[coord] for coord in [(0 + offset,0 + offset), (2 + offset,2 + offset)]]
        if physical_reset_op == "reset_y":
            circ = model.h_gate(h_gates + h_0) + model.idle(set(qubits)-set(h_gates + h_0))
            circ += model.s_gate([l[(1 + offset, 1 + offset)]])
            yield circ
        else:
            yield model.h_gate(h_gates + h_0) + model.idle(set(qubits)-set(h_gates + h_0))
        yield model.tick()
        # Step 1
        czs_1 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,0 + offset),
        (2 + offset,0 + offset),(2 + offset,2 + offset)]]

        yield model.cz(czs_1) + model.idle(set(qubits)-set(czs_1))
        yield model.tick()
        # Step 2
        czs_2 = [l[coord] for coord in [(0 + offset,2 + offset),(0 + offset,0 + offset)
        ,(1 + offset,1 + offset),(2 + offset,2 + offset)]]
        h_2 = [l[coord] for coord in [(0 + offset, 0 + offset),(1 + offset, 1 + offset),(2 + offset, 2 + offset)]]
        yield model.cz(czs_2) + model.idle(set(qubits)-set(czs_2))
        yield model.tick()
        yield model.h_gate(h_2) + model.idle(set(qubits)-set(h_2))
        yield model.tick()
        # Step 3
        czs_3 = [l[coord] for coord in [(0 + offset,2 + offset),(1 + offset,1 + offset)]]
        yield model.cz(czs_3) + model.idle(set(qubits)-set(czs_3))
        yield model.tick()
        # Step 4
        czs_4 = [l[coord] for coord in [(2 + offset,0 + offset),(1 + offset,1 + offset)]]
        h_4 = [l[coord] for coord in [(1 + offset,1 + offset)]]
        yield model.cz(czs_4) + model.idle(set(qubits)-set(czs_4))
        yield model.tick()
        yield model.h_gate(h_4) + model.idle(set(qubits)-set(h_4))
        yield model.tick()

        if d >= 4:
            offset = d-4
            # H gate
            h_gates = [l[coord] for coord in [(2 + offset,0 + offset), (4 + offset,0 + offset),
            (1 + offset,1 + offset), (5 + offset,1 + offset),
            (1 + offset,3 + offset), (5 + offset,3 + offset),
            (1 + offset,5 + offset), (5 + offset,5 + offset),
            (2 + offset,6 + offset), (4 + offset,6 + offset)]]
            # decompose CNOT into H CZ H
            h_0 = [l[coord] for coord in [(0 + offset,0 + offset), (6 + offset,0 + offset),
            (0 + offset,2 + offset), (4 + offset,2 + offset),
            (1 + offset,3 + offset), (2 + offset,4 + offset),
            (6 + offset,4 + offset), (0 + offset,6 + offset),
            (6 + offset,6 + offset)]]
            yield model.h_gate(h_gates + h_0) + model.idle(set(qubits)-set(h_gates + h_0))
            yield model.tick()
            # Step 1
            czs_1 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,0 + offset),
            (5 + offset,1 + offset),(6 + offset,0 + offset),
            (2 + offset,2 + offset),(0 + offset,2 + offset),
            (3 + offset,1 + offset),(4 + offset,2 + offset),
            (3 + offset,3 + offset),(1 + offset,3 + offset),
            (3 + offset,5 + offset),(2 + offset,4 + offset),
            (4 + offset,4 + offset),(6 + offset,4 + offset),
            (1 + offset,5 + offset),(0 + offset,6 + offset),
            (5 + offset,5 + offset),(6 + offset,6 + offset)]]
            h_1 = [l[coord] for coord in [(0 + offset, 0 + offset),(6 + offset, 0 + offset),(1 + offset, 3 + offset),
            (0 + offset, 6 + offset),(6 + offset, 6 + offset),
            (3 + offset, 1 + offset),(2 + offset, 2 + offset),
            (6 + offset, 2 + offset),(5 + offset, 3 + offset),
            (0 + offset, 4 + offset),(4 + offset, 4 + offset),(3 + offset, 5 + offset),
        (2 + offset, 4 + offset),(4 + offset, 2 + offset)]]
            yield model.cz(czs_1) + model.idle(set(qubits)-set(czs_1))
            yield model.tick()
            yield model.h_gate(h_1) + model.idle(set(qubits)-set(h_1))
            yield model.tick()
            # Step 2
            czs_2 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,2 + offset),
            (2 + offset,0 + offset),(2 + offset,2 + offset),
            (4 + offset,0 + offset),(3 + offset,1 + offset),
            (4 + offset,2 + offset),(6 + offset,2 + offset),
            (3 + offset,3 + offset),(5 + offset,3 + offset),
            (2 + offset,4 + offset),(0 + offset,4 + offset),
            (2 + offset,6 + offset),(3 + offset,5 + offset),
            (4 + offset,6 + offset),(4 + offset,4 + offset),
            (5 + offset,5 + offset),(6 + offset,4 + offset)]]
            h_2 = [l[coord] for coord in [(2 + offset, 2 + offset),(5 + offset, 3 + offset),(4 + offset, 4 + offset),
        (2 + offset, 4 + offset),(4 + offset, 2 + offset)]]
            yield model.cz(czs_2) + model.idle(set(qubits)-set(czs_2))
            yield model.tick()
            yield model.h_gate(h_2) + model.idle(set(qubits)-set(h_2))
            yield model.tick()
            # Step 3
            czs_3 = [l[coord] for coord in [(2 + offset,0 + offset),(3 + offset,1 + offset),
            (4 + offset,0 + offset),(4 + offset,2 + offset),
            (5 + offset,1 + offset),(6 + offset,2 + offset),
            (1 + offset,3 + offset),(0 + offset,2 + offset),
            (5 + offset,3 + offset),(6 + offset,4 + offset),
            (1 + offset,5 + offset),(0 + offset,4 + offset),
            (2 + offset,6 + offset),(2 + offset,4 + offset),
            (4 + offset,6 + offset),(3 + offset,5 + offset)]]
            h_3 = [l[coord] for coord in [(3 + offset, 1 + offset),(4 + offset, 2 + offset),(0 + offset, 2 + offset),
            (6 + offset, 4 + offset),(2 + offset, 4 + offset),(3 + offset, 5 + offset),
            (1 + offset, 1 + offset),(5 + offset, 1 + offset),(1 + offset, 5 + offset),(5 + offset, 5 + offset)]]
            yield model.cz(czs_3) + model.idle(set(qubits)-set(czs_3))
            yield model.tick()
            yield model.h_gate(h_3) + model.idle(set(qubits)-set(h_3))
            yield model.tick()
            # Step 4
            czs_4 = [l[coord] for coord in [(2 + offset,0 + offset),(1 + offset,1 + offset),
            (4 + offset,0 + offset),(5 + offset,1 + offset),
            (1 + offset,3 + offset),(0 + offset,4 + offset),
            (5 + offset,3 + offset),(6 + offset,2 + offset),
            (2 + offset,6 + offset),(1 + offset,5 + offset),
            (4 + offset,6 + offset),(5 + offset,5 + offset)]]
            h_4 = [l[coord] for coord in [(1 + offset,1 + offset),(5 + offset,1 + offset),
            (6 + offset,2 + offset),(0 + offset,4 + offset),
            (1 + offset,5 + offset),(5 + offset,5 + offset)]]
            yield model.cz(czs_4) + model.idle(set(qubits)-set(czs_4))
            yield model.tick()
            yield model.h_gate(h_4) + model.idle(set(qubits)-set(h_4))
            yield model.tick()
    else:
        init_x = [l[coord] for coord in [(2 + offset,0 + offset), (0 + offset,2 + offset)]]
        circ = model.reset_x(init_x) + model.reset(set(qubits)-set(init_x))
        if physical_reset_op == "reset_x":
            circ += model.reset_x([l[(1 + offset, 1 + offset)]])
        elif physical_reset_op == "reset_y":
            circ += model.reset_y([l[(1 + offset, 1 + offset)]])
        yield circ
        yield model.tick()
        # Step 1
        cnots_1 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,0 + offset),
        (2 + offset,0 + offset),(2 + offset,2 + offset)]]
        yield model.cnot(cnots_1) + model.idle(set(qubits)-set(cnots_1))
        yield model.tick()
        # Step 2
        cnots_2 = [l[coord] for coord in [(0 + offset,2 + offset),(0 + offset,0 + offset)
        ,(1 + offset,1 + offset),(2 + offset,2 + offset)]]
        yield model.cnot(cnots_2) + model.idle(set(qubits)-set(cnots_2))
        yield model.tick()
        # Step 3
        cnots_3 = [l[coord] for coord in [(0 + offset,2 + offset),(1 + offset,1 + offset)]]
        yield model.cnot(cnots_3) + model.idle(set(qubits)-set(cnots_3))
        yield model.tick()
        # Step 4
        cnots_4 = [l[coord] for coord in [(2 + offset,0 + offset),(1 + offset,1 + offset)]]
        yield model.cnot(cnots_4) + model.idle(set(qubits)-set(cnots_4))
        yield model.tick()


        if d >= 4:
            offset = d-4
            # H gate
            init_x = [l[coord] for coord in [(2 + offset,0 + offset), (4 + offset,0 + offset),
            (1 + offset,1 + offset), (5 + offset,1 + offset),
            (1 + offset,3 + offset), (5 + offset,3 + offset),
            (1 + offset,5 + offset), (5 + offset,5 + offset),
            (2 + offset,6 + offset), (4 + offset,6 + offset)]]
            yield model.reset_x(init_x) + model.idle(set(qubits)-set(init_x))
            yield model.tick()
            # Step 1
            cnots_1 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,0 + offset),
            (5 + offset,1 + offset),(6 + offset,0 + offset),
            (2 + offset,2 + offset),(0 + offset,2 + offset),
            (3 + offset,1 + offset),(4 + offset,2 + offset),
            (3 + offset,3 + offset),(1 + offset,3 + offset),
            (3 + offset,5 + offset),(2 + offset,4 + offset),
            (4 + offset,4 + offset),(6 + offset,4 + offset),
            (1 + offset,5 + offset),(0 + offset,6 + offset),
            (5 + offset,5 + offset),(6 + offset,6 + offset)]]
            yield model.cnot(cnots_1) + model.idle(set(qubits)-set(cnots_1))
            yield model.tick()
            # Step 2
            cnots_2 = [l[coord] for coord in [(1 + offset,1 + offset),(0 + offset,2 + offset),
            (2 + offset,0 + offset),(2 + offset,2 + offset),
            (4 + offset,0 + offset),(3 + offset,1 + offset),
            (4 + offset,2 + offset),(6 + offset,2 + offset),
            (3 + offset,3 + offset),(5 + offset,3 + offset),
            (2 + offset,4 + offset),(0 + offset,4 + offset),
            (2 + offset,6 + offset),(3 + offset,5 + offset),
            (4 + offset,6 + offset),(4 + offset,4 + offset),
            (5 + offset,5 + offset),(6 + offset,4 + offset)]]
            yield model.cnot(cnots_2) + model.idle(set(qubits)-set(cnots_2))
            yield model.tick()
            # Step 3
            cnots_3 = [l[coord] for coord in [(2 + offset,0 + offset),(3 + offset,1 + offset),
            (4 + offset,0 + offset),(4 + offset,2 + offset),
            (5 + offset,1 + offset),(6 + offset,2 + offset),
            (1 + offset,3 + offset),(0 + offset,2 + offset),
            (5 + offset,3 + offset),(6 + offset,4 + offset),
            (1 + offset,5 + offset),(0 + offset,4 + offset),
            (2 + offset,6 + offset),(2 + offset,4 + offset),
            (4 + offset,6 + offset),(3 + offset,5 + offset)]]
            yield model.cnot(cnots_3) + model.idle(set(qubits)-set(cnots_3))
            yield model.tick()
            # Step 4
            cnots_4 = [l[coord] for coord in [(2 + offset,0 + offset),(1 + offset,1 + offset),
            (4 + offset,0 + offset),(5 + offset,1 + offset),
            (1 + offset,3 + offset),(0 + offset,4 + offset),
            (5 + offset,3 + offset),(6 + offset,2 + offset),
            (2 + offset,6 + offset),(1 + offset,5 + offset),
            (4 + offset,6 + offset),(5 + offset,5 + offset)]]
            yield model.cnot(cnots_4) + model.idle(set(qubits)-set(cnots_4))
            yield model.tick()
    # grow code to maximum size
    for d in range(4, layout.distance, 2):
        yield from _grow_code_iterator(
            model=model, layout=layout, curr_distance=d, primitive_gates=primitive_gates
        )




def _grow_code_iterator(
    model: Model,
    layout: Layout,
    curr_distance: int,
    primitive_gates: str,
) -> Generator[Circuit]:
    """
    Yields stim circuit blocks which as a whole correspond to an circuit growing
    a distance ``d`` unrotated surface code to a ``d+2`` one of the given model
    without the detectors. Note that this circuit is not fault tolerant.

    Parameters
    ----------
    model
        Noise model for the gates.
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
    if curr_distance % 2 != layout.distance % 2:
        raise TypeError(f"'curr_distance' and 'layout_distance' must have the same parity, but curr_distance:{curr_distance}, layout.distance:{layout.distance} were given.")
    if curr_distance >= layout.distance:
        raise TypeError(
            f"'curr_distance' ({curr_distance}) must be strictly smaller than "
            f"the code distance ({layout.distance})."
        )
    if primitive_gates not in ["cnot", "cz"]:
        raise ValueError(f"'{primitive_gates}' is not available as primitive gate set.")

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    # maps from coordinates to qubit labels
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
    # l: dict[tuple[int, int], str] = {}
    # for qubit, coord in layout.qubit_coords.items():
    #     l[(int(coord[0]), int(coord[1]))] = qubit

    d = layout.distance
    target_d = curr_distance + 2
    offset = d - target_d
    qubits = set(layout.qubits)

    if primitive_gates == "cz":
        # initial |+> state + decompose CNOT into H CZ H
        # h at 4 corners
        h_gates = [l[coord] for coord in [(1+offset,1+offset),
                    (2*target_d-3+offset,1+offset),
                    (1+offset,2*target_d-3+offset),
                    (2*target_d-3+offset,2*target_d-3+offset)]] 
        for i in range(2, 2*target_d-2, 2):
            h_gates += [l[(i+offset,offset)]] #top
            h_gates += [l[(i+offset,2*target_d-2+offset)]] #bottom
        for i in range(0, 2*target_d, 2):
            h_gates += [l[(0+offset,i+offset)]] #left
            h_gates += [l[(2*target_d-2+offset,i+offset)]] #right
        yield model.h_gate(h_gates) + model.idle(set(qubits)-set(h_gates))
        yield model.tick()
        # first time step
        # czs at 4 corner
        czs_1 = [l[coord] for coord in [(1+offset,1+offset),(offset,offset)
                    ,(2*target_d-3+offset,1+offset),(2*target_d-2+offset,offset)
                    ,(1+offset,2*target_d-3+offset),(offset,2*target_d-2+offset)
                    ,(2*target_d-3+offset,2*target_d-3+offset),(2*target_d-2+offset,2*target_d-2+offset)]]
        # czs at outer line
        for i in range(2, 2*target_d-2, 2):
            czs_1 += [l[(2+offset,i+offset)],l[(offset,i+offset)]] #left
            czs_1 += [l[(2*target_d-4+offset,i+offset)],l[(2*target_d-2+offset,i+offset)]] #right
        # czs at inner corner
        for i in range(3, 2*target_d-3, 2):
            czs_1 += [l[(3+offset,i+offset)],l[(1+offset,i+offset)]] #left
            czs_1 += [l[(2*target_d-5+offset,i+offset)],l[(2*target_d-3+offset,i+offset)]] #right
        # h at 4 corner
        h_1 = [l[coord] for coord in [(0+offset,0+offset),
                (2*target_d-2+offset,offset),
                (offset,2*target_d-2+offset),
                (2*target_d-2+offset,2*target_d-2+offset)]]
        # h at outer line
        for i in range(2, 2*target_d-2, 2):
            h_1 += [l[(i+offset,2+offset)]] #top
            h_1 += [l[(i+offset,2*target_d-4+offset)]] #bottom
        # h at inner corner
        for i in range(3, 2*target_d-3, 2):
            h_1 += [l[(i+offset,3+offset)]] #top
            h_1 += [l[(i+offset,2*target_d-5+offset)]] #bottom
            h_1 += [l[(1+offset,i+offset)]] #left 
            h_1 += [l[(2*target_d-3+offset,i+offset)]] #right
        yield model.cz(czs_1) + model.idle(set(qubits)-set(czs_1))
        yield model.tick()
        yield model.h_gate(h_1) + model.idle(set(qubits)-set(h_1))
        yield model.tick()
        # second time step
        # cnots at 4 corner
        czs_2 = [l[coord] for coord in [(1+offset,1+offset),(offset,2+offset)
                    ,(2*target_d-3+offset,1+offset),(2*target_d-2+offset,offset+2)
                    ,(1+offset,2*target_d-3+offset),(offset,2*target_d-4+offset)
                    ,(2*target_d-3+offset,2*target_d-3+offset),(2*target_d-2+offset,2*target_d-4+offset)]]
        # cnots at outer line
        for i in range(2, 2*target_d-2, 2):
            czs_2 += [l[(i+offset,offset)],l[(i+offset,2+offset)]] #top cnot
            czs_2 += [l[(i+offset,2*target_d-2+offset)],l[(i+offset,2*target_d-4+offset)]] #bottom cnot
        # cnots at inner corner
        for i in range(3, 2*target_d-3, 2):
            czs_2 += [l[(i+offset,1+offset)],l[(i+offset,3+offset)]] #top cnot
            czs_2 += [l[(i+offset,2*target_d-3+offset)],l[(i+offset,2*target_d-5+offset)]] #bottom cnot       
        # h at outer line
        h_2 = []
        for i in range(2, 2*target_d-2, 2):
            h_2 += [l[(i+offset,2+offset)]] #top
            h_2 += [l[(i+offset,2*target_d-4+offset)]] #bottom
            h_2 += [l[(i+1+offset,1+offset)]] #top, next step
            h_2 += [l[(i-1+offset,2*target_d-3+offset)]] #bottom, next step
        # h at inner corner
        for i in range(3, 2*target_d-3, 2):
            h_2 += [l[(i+offset,3+offset)]] #top
            h_2 += [l[(i+offset,2*target_d-5+offset)]] #bottom
        yield model.cz(czs_2) + model.idle(set(qubits)-set(czs_2))
        yield model.tick()
        yield model.h_gate(h_2) + model.idle(set(qubits)-set(h_2))
        yield model.tick()
        #third time step
        czs_3 = []
        for i in range(2, 2*target_d-2,2):
            czs_3 += [l[(i+offset,offset)],l[(i+1+offset,1+offset)]] #top
            czs_3 += [l[(i+offset,2*target_d-2+offset)],l[(i-1+offset,2*target_d-3+offset)]] #bottom
        for i in range(3, 2*target_d-3, 2):
            czs_3 += [l[(1+offset,i+offset)],l[(offset,i-1+offset)]] #left
            czs_3 += [l[(2*target_d-3+offset,i+offset)],l[(2*target_d-2+offset,i+1+offset)]] #right
        # h 6 corners
        h_3 = [l[coord] for coord in [(1+offset,1+offset),
                (0+offset,2+offset),
                (2*target_d-3+offset,1+offset),
                (2*target_d-3+offset,2*target_d-3+offset),
                (2*target_d-2+offset,2*target_d-4+offset),
                (1+offset,2*target_d-3+offset)]]
        yield model.cz(czs_3) + model.idle(set(qubits)-set(czs_3))
        yield model.tick()
        yield model.h_gate(h_3) + model.idle(set(qubits)-set(h_3))
        yield model.tick()
        #fourth time step
        czs_4 = []
        for i in range(2, 2*target_d-2,2):
            czs_4 += [l[(i+offset,offset)],l[(i-1+offset,1+offset)]] #top
            czs_4 += [l[(i+offset,2*target_d-2+offset)],l[(i+1+offset,2*target_d-3+offset)]] #bottom
        for i in range(3, 2*target_d-3, 2):
            czs_4 += [l[(1+offset,i+offset)],l[(offset,i+1+offset)]] #left
            czs_4 += [l[(2*target_d-3+offset,i+offset)],l[(2*target_d-2+offset,i-1+offset)]] #right
        h_4 = []
        for i in range(2, 2*target_d-2,2):
            h_4 += [l[(i-1+offset,1+offset)]] #top
            h_4 += [l[(i+1+offset,2*target_d-3+offset)]] #bottom
        for i in range(3, 2*target_d-3, 2):
            h_4 += [l[(offset,i+1+offset)]] #left
            h_4 += [l[(2*target_d-2+offset,i-1+offset)]] #right 
        yield model.cz(czs_4) + model.idle(set(qubits)-set(czs_4))
        yield model.tick()
        yield model.h_gate(h_4) + model.idle(set(qubits)-set(h_4))
        yield model.tick()
    else:
        h_gates = [] 
        for i in range(2, 2*target_d-2, 2):
            h_gates += [l[(i+offset,offset)]] #top
            h_gates += [l[(i+offset,2*target_d-2+offset)]] #bottom
        for i in range(1, 2*target_d-1, 2):
            h_gates += [l[(1+offset,i+offset)]] #left
            h_gates += [l[(2*target_d-3+offset,i+offset)]] #right
        yield model.h_gate(h_gates) + model.idle(set(qubits)-set(h_gates))
        yield model.tick()
        # first time step
        # cnots at 4 corner
        cnots_1 = [l[coord] for coord in [(1+offset,1+offset),(offset,offset)
                    ,(2*target_d-3+offset,1+offset),(2*target_d-2+offset,offset)
                    ,(1+offset,2*target_d-3+offset),(offset,2*target_d-2+offset)
                    ,(2*target_d-3+offset,2*target_d-3+offset),(2*target_d-2+offset,2*target_d-2+offset)]]
        # cnots at outer line
        for i in range(2, 2*target_d-2, 2):
            cnots_1 += [l[(2+offset,i+offset)],l[(offset,i+offset)]] #left
            cnots_1 += [l[(2*target_d-4+offset,i+offset)],l[(2*target_d-2+offset,i+offset)]] #right
        # cnots at inner corner
        for i in range(3, 2*target_d-3, 2):
            cnots_1 += [l[(3+offset,i+offset)],l[(1+offset,i+offset)]] #left
            cnots_1 += [l[(2*target_d-5+offset,i+offset)],l[(2*target_d-3+offset,i+offset)]] #right
        yield model.cnot(cnots_1) + model.idle(set(qubits)-set(cnots_1))
        yield model.tick()
        # second time step
        # cnots at 4 corner
        cnots_2 = [l[coord] for coord in [(1+offset,1+offset),(offset,2+offset)
                    ,(2*target_d-3+offset,1+offset),(2*target_d-2+offset,offset+2)
                    ,(1+offset,2*target_d-3+offset),(offset,2*target_d-4+offset)
                    ,(2*target_d-3+offset,2*target_d-3+offset),(2*target_d-2+offset,2*target_d-4+offset)]]
        # cnots at outer line
        for i in range(2, 2*target_d-2, 2):
            cnots_2 += [l[(i+offset,offset)],l[(i+offset,2+offset)]] #top cnot
            cnots_2 += [l[(i+offset,2*target_d-2+offset)],l[(i+offset,2*target_d-4+offset)]] #bottom cnot
        # cnots at inner corner
        for i in range(3, 2*target_d-3, 2):
            cnots_2 += [l[(i+offset,1+offset)],l[(i+offset,3+offset)]] #top cnot
            cnots_2 += [l[(i+offset,2*target_d-3+offset)],l[(i+offset,2*target_d-5+offset)]] #bottom cnot       
        yield model.cnot(cnots_2) + model.idle(set(qubits)-set(cnots_2))
        yield model.tick()
        #third time step
        cnots_3 = []
        for i in range(2, 2*target_d-2,2):
            cnots_3 += [l[(i+offset,offset)],l[(i+1+offset,1+offset)]] #top
            cnots_3 += [l[(i+offset,2*target_d-2+offset)],l[(i-1+offset,2*target_d-3+offset)]] #bottom
        for i in range(3, 2*target_d-3, 2):
            cnots_3 += [l[(1+offset,i+offset)],l[(offset,i-1+offset)]] #left
            cnots_3 += [l[(2*target_d-3+offset,i+offset)],l[(2*target_d-2+offset,i+1+offset)]] #right
        yield model.cnot(cnots_3) + model.idle(set(qubits)-set(cnots_3))
        yield model.tick()
        #fourth time step
        cnots_4 = []
        for i in range(2, 2*target_d-2,2):
            cnots_4 += [l[(i+offset,offset)],l[(i-1+offset,1+offset)]] #top
            cnots_4 += [l[(i+offset,2*target_d-2+offset)],l[(i+1+offset,2*target_d-3+offset)]] #bottom
        for i in range(3, 2*target_d-3, 2):
            cnots_4 += [l[(1+offset,i+offset)],l[(offset,i+1+offset)]] #left
            cnots_4 += [l[(2*target_d-3+offset,i+offset)],l[(2*target_d-2+offset,i-1+offset)]] #right
        yield model.cnot(cnots_4) + model.idle(set(qubits)-set(cnots_4))
        yield model.tick()

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
    The implementation follows Figure 2, 9 from:

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
    The implementation follows Figure 2, 9 from:

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
    The implementation follows Figure 2, 9 from:

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
    The implementation follows Figure 2, 9 from:

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
    The implementation follows Figure 2, 9 from:

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
    The implementation follows Figure 2, 9 from:

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
    The implementation follows Figure 2, 9 from:

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

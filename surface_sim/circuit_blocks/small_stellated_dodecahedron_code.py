from collections.abc import Iterator, Sequence
from itertools import chain

from stim import Circuit

from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors, get_new_stab_dict_from_layout
from .decorators import qec_circuit, sq_gate

# methods to have in this script
from .util import (
    qubit_coords,
    idle_iterator,
    log_meas,
    log_meas_iterator,
    log_meas_z_iterator,
    log_meas_x_iterator,
    init_qubits,
    init_qubits_iterator,
    init_qubits_z0_iterator,
    init_qubits_z1_iterator,
    init_qubits_x0_iterator,
    init_qubits_x1_iterator,
)

__all__ = [
    "qubit_coords",
    "idle_iterator",
    "log_meas",
    "log_meas_iterator",
    "log_meas_z_iterator",
    "log_meas_x_iterator",
    "init_qubits",
    "init_qubits_iterator",
    "init_qubits_z0_iterator",
    "init_qubits_z1_iterator",
    "init_qubits_x0_iterator",
    "init_qubits_x1_iterator",
    "log_fold_trans_h",
    "log_fold_trans_h_iterator",
    "log_fold_trans_s",
    "log_fold_trans_s_iterator",
    "log_fold_trans_swap_r",
    "log_fold_trans_swap_r_iterator",
    "log_fold_trans_swap_s",
    "log_fold_trans_swap_s_iterator",
    "qec_round",
    "qec_round_iterator",
    "gate_to_iterator",
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
    This implementation follows the interaction order specified in the layout.
    This implementation uses CNOTs, and resets and measurements in the X basis.
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
    This implementation follows the interaction order specified in the layout.
    This implementation uses CNOTs, and resets and measurements in the X basis.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not an small stellated dodecahedron code, "
            f"but a {layout.code}"
        )
    if not anc_reset:
        raise ValueError("Current implementation only supports 'anc_reset=True'.")

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    qubits = set(layout.qubits)

    int_order = layout.interaction_order
    num_steps = len(int_order[anc_qubits[0]])
    x_stabs = layout.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout.get_qubits(role="anc", stab_type="z_type")

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if anc_reset:
        resets = model.reset_x(x_stabs) + model.reset_z(z_stabs)
        yield resets + model.idle(data_qubits)
        yield model.tick()

    # CNOT gates
    for step in range(num_steps):
        cnot_circuit = Circuit()
        interacted_qubits = set()

        # X ancillas
        int_pairs = [(x, int_order[x][step]) for x in x_stabs]
        int_pairs = [pair for pair in int_pairs if pair[1] is not None]
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)
        cnot_circuit += model.cnot(int_qubits)

        # Z ancillas
        int_pairs = [(int_order[z][step], z) for z in z_stabs]
        int_pairs = [pair for pair in int_pairs if pair[0] is not None]
        int_qubits = list(chain.from_iterable(int_pairs))
        interacted_qubits.update(int_qubits)
        cnot_circuit += model.cnot(int_qubits)

        idle_qubits = qubits - interacted_qubits
        yield cnot_circuit + model.idle(idle_qubits)
        yield model.tick()

    meas = model.measure_x(x_stabs) + model.measure_z(z_stabs)
    yield meas + model.idle(data_qubits)
    yield model.tick()


def log_fold_trans_s(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical S-like gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_s"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(log_fold_trans_s_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_fold_trans_s_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical S-like gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    gate_label = "log_fold_trans_s"

    cz_pairs = set()
    qubits_s_gate = set()
    qubits_s_dag_gate = set()
    for data_qubit in data_qubits:
        trans_s = layout.param(gate_label, data_qubit)
        if trans_s is None:
            raise ValueError(
                "The layout does not have the information to run a "
                f"transversal S-like gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        # Using a set to avoid repetition of the cz gates.
        # Using tuple so that the object is hashable for the set.
        if trans_s["cz"] is not None:
            cz_pairs.add(tuple(sorted([data_qubit, trans_s["cz"]])))
        if trans_s["local"] == "S":
            qubits_s_gate.add(data_qubit)
        elif trans_s["local"] == "S_DAG":
            qubits_s_dag_gate.add(data_qubit)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # S, S_DAG gates
    int_qubits = list(chain.from_iterable(cz_pairs))
    yield (
        model.s_gate(qubits_s_gate)
        + model.s_dag_gate(qubits_s_dag_gate)
        + model.cphase(int_qubits)
        + model.idle(anc_qubits)
    )
    yield model.tick()


def log_fold_trans_h(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical H-like gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_h"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(log_fold_trans_h_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_fold_trans_h_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical H-like gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = "log_fold_trans_h"

    swap_pairs = set()
    qubits_h_gate = set()
    for data_qubit in data_qubits:
        trans_h = layout.param(gate_label, data_qubit)
        if trans_h is None:
            raise ValueError(
                "The layout does not have the information to run a "
                f"transversal H gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        # Using a set to avoid repetition of the swap gates.
        # Using tuple so that the object is hashable for the set.
        if trans_h["swap"] is not None:
            swap_pairs.add(tuple(sorted([data_qubit, trans_h["swap"]])))
        if trans_h["local"] == "H":
            qubits_h_gate.add(data_qubit)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # H gates
    idle_qubits = qubits - qubits_h_gate
    yield model.hadamard(qubits_h_gate) + model.idle(idle_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.swap(int_qubits) + model.idle(idle_qubits)
    yield model.tick()


def log_fold_trans_swap_r(
    model: Model, layout: Layout, detectors: Detectors
) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_r` gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_swap_r"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(
        log_fold_trans_swap_r_iterator(model=model, layout=layout), start=Circuit()
    )


@sq_gate
def log_fold_trans_swap_r_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_r` gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = "log_fold_trans_swap_r"

    swap_pairs = set()
    for data_qubit in data_qubits:
        trans_swap = layout.param(gate_label, data_qubit)
        if trans_swap is None:
            raise ValueError(
                "The layout does not have the information to run a "
                f"transversal SWAP gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        # Using a set to avoid repetition of the swap gates.
        # Using tuple so that the object is hashable for the set.
        if trans_swap["swap"] is not None:
            swap_pairs.add(tuple(sorted([data_qubit, trans_swap["swap"]])))

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.swap(int_qubits) + model.idle(idle_qubits)
    yield model.tick()


def log_fold_trans_swap_s(
    model: Model, layout: Layout, detectors: Detectors
) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_s` gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_swap_s"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(
        log_fold_trans_swap_s_iterator(model=model, layout=layout), start=Circuit()
    )


@sq_gate
def log_fold_trans_swap_s_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_s` gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = "log_fold_trans_swap_s"

    swap_pairs = set()
    for data_qubit in data_qubits:
        trans_swap = layout.param(gate_label, data_qubit)
        if trans_swap is None:
            raise ValueError(
                "The layout does not have the information to run a "
                f"transversal SWAP gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        # Using a set to avoid repetition of the swap gates.
        # Using tuple so that the object is hashable for the set.
        if trans_swap["swap"] is not None:
            swap_pairs.add(tuple(sorted([data_qubit, trans_swap["swap"]])))

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.swap(int_qubits) + model.idle(idle_qubits)
    yield model.tick()


gate_to_iterator = {
    "TICK": qec_round_iterator,
    "I": idle_iterator,
    "R": init_qubits_z0_iterator,
    "RZ": init_qubits_z0_iterator,
    "RX": init_qubits_x0_iterator,
    "M": log_meas_z_iterator,
    "MZ": log_meas_z_iterator,
    "MX": log_meas_x_iterator,
}

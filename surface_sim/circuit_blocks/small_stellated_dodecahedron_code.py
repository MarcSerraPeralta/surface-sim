from collections.abc import Collection, Generator
from itertools import chain

from stim import Circuit

from ..detectors import Detectors, get_new_stab_dict_from_layout
from ..layouts.layout import Layout
from ..models import Model
from .decorators import qec_circuit, qubit_init_x, qubit_init_z, sq_gate
from .util import general_qec_round_iterator_cnots as general_qec_round_iterator

# methods to have in this script
from .util import (
    idle_iterator,
    log_depolarize1_error_iterator,
    log_meas,
    log_meas_iterator,
    log_meas_x_iterator,
    log_meas_z_iterator,
    log_x_error_iterator,
    log_y_error_iterator,
    log_z_error_iterator,
    qubit_coords,
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
    "log_fold_trans_swap_a",
    "log_fold_trans_swap_a_iterator",
    "log_fold_trans_swap_b",
    "log_fold_trans_swap_b_iterator",
    "log_fold_trans_swap_c",
    "log_fold_trans_swap_c_iterator",
    "log_x_error_iterator",
    "log_y_error_iterator",
    "log_z_error_iterator",
    "log_depolarize1_error_iterator",
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
    This implementation follows the interaction order specified in the layout.
    This implementation uses CNOTs, and resets and measurements in the X basis.

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
    This implementation follows the interaction order specified in the layout.
    This implementation uses CNOTs, and resets and measurements in the X basis.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not an small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    yield from general_qec_round_iterator(
        model=model, layout=layout, anc_reset=anc_reset
    )


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
def log_fold_trans_s_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
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
def log_fold_trans_h_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
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
def log_fold_trans_swap_r_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
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

    swap_1_pairs = set()
    swap_2_pairs = set()
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
        if trans_swap["swap_1"] is not None:
            swap_1_pairs.add(tuple(sorted([data_qubit, trans_swap["swap_1"]])))
        if trans_swap["swap_2"] is not None:
            swap_2_pairs.add(tuple(sorted([data_qubit, trans_swap["swap_2"]])))

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_1_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.swap(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_2_pairs))
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
def log_fold_trans_swap_s_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
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

    swap_1_pairs = set()
    swap_2_pairs = set()
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
        if trans_swap["swap_1"] is not None:
            swap_1_pairs.add(tuple(sorted([data_qubit, trans_swap["swap_1"]])))
        if trans_swap["swap_2"] is not None:
            swap_2_pairs.add(tuple(sorted([data_qubit, trans_swap["swap_2"]])))

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_1_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.swap(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_2_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.swap(int_qubits) + model.idle(idle_qubits)
    yield model.tick()


def log_fold_trans_swap_a(
    model: Model, layout: Layout, detectors: Detectors
) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_a` gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_swap_a"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(
        log_fold_trans_swap_a_iterator(model=model, layout=layout), start=Circuit()
    )


@sq_gate
def log_fold_trans_swap_a_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_a` gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = "log_fold_trans_swap_a"

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


def log_fold_trans_swap_b(
    model: Model, layout: Layout, detectors: Detectors
) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_b` gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_swap_a"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(
        log_fold_trans_swap_b_iterator(model=model, layout=layout), start=Circuit()
    )


@sq_gate
def log_fold_trans_swap_b_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_b` gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = "log_fold_trans_swap_b"

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


def log_fold_trans_swap_c(
    model: Model, layout: Layout, detectors: Detectors
) -> Circuit:
    """
    Returns the stim circuit corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_c` gate.
    See the corresponding setting function for more information.
    """
    # update the stabilizer generators
    gate_label = "log_fold_trans_swap_c"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(
        log_fold_trans_swap_c_iterator(model=model, layout=layout), start=Circuit()
    )


@sq_gate
def log_fold_trans_swap_c_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields the stim circuits corresponding to a transversal logical SWAP-like gate,
    in particular the :math:`\\sigma_c` gate.
    See the corresponding setting function for more information.
    """
    if layout.code != "small_stellated_dodecahedron_code":
        raise TypeError(
            "The given layout is not a small stellated dodecahedron code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = "log_fold_trans_swap_c"

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


def init_qubits(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    data_init: dict[str, int],
    rot_basis: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical initialization
    of the given model.
    The ancilla qubits are also initialized in the correct basis for the
    case ``anc_reset = False``.

    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    # activate detectors
    # the order of activating the detectors or applying the circuit
    # does not matter because this will be done in a layer of logical operations,
    # so no QEC round are run simultaneously
    anc_qubits = layout.anc_qubits
    stab_type = "z_type" if rot_basis else "x_type"
    gauge_dets = layout.get_qubits(role="anc", stab_type=stab_type)
    detectors.activate_detectors(anc_qubits, gauge_dets=gauge_dets)
    return sum(
        init_qubits_iterator(
            model=model,
            layout=layout,
            data_init=data_init,
            rot_basis=rot_basis,
        ),
        start=Circuit(),
    )


def init_qubits_iterator(
    model: Model,
    layout: Layout,
    data_init: dict[str, int],
    rot_basis: bool = False,
) -> Generator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization
    of the given model.
    The ancilla qubits are also initialized in the correct basis for the
    case ``anc_reset = False``.

    By default, the logical initialization is in the Z basis.
    If rot_basis, the logical initialization is in the X basis.
    """
    xancs = layout.get_qubits(role="anc", stab_type="x_type")
    zancs = layout.get_qubits(role="anc", stab_type="z_type")
    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)

    reset_data = model.reset_x(data_qubits) if rot_basis else model.reset(data_qubits)

    yield reset_data + model.reset(zancs) + model.reset_x(xancs)
    yield model.tick()

    init_circ = Circuit()
    exc_qubits = set([q for q, s in data_init.items() if s and (q in data_qubits)])
    if exc_qubits:
        if rot_basis:
            init_circ += model.z_gate(exc_qubits)
        else:
            init_circ += model.x_gate(exc_qubits)

    idle_qubits = qubits - exc_qubits
    yield init_circ + model.idle(idle_qubits)
    yield model.tick()


@qubit_init_z
def init_qubits_z0_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |0>
    state of the given model.
    The ancilla qubits are also initialized in the correct basis for the
    case ``anc_reset = False``.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 0 for q in layout.data_qubits}
    yield from init_qubits_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=False
    )


@qubit_init_z
def init_qubits_z1_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |1>
    state of the given model.
    The ancilla qubits are also initialized in the correct basis for the
    case ``anc_reset = False``.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 1 for q in layout.data_qubits}
    if layout.num_data_qubits % 2 == 0:
        # ensure that the bistring corresponds to the |1> state
        data_init[layout.data_qubits[-1]] = 0

    yield from init_qubits_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=False
    )


@qubit_init_x
def init_qubits_x0_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |+>
    state of the given model.
    The ancilla qubits are also initialized in the correct basis for the
    case ``anc_reset = False``.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 0 for q in layout.data_qubits}
    yield from init_qubits_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=True
    )


@qubit_init_x
def init_qubits_x1_iterator(model: Model, layout: Layout) -> Generator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |->
    state of the given model.
    The ancilla qubits are also initialized in the correct basis for the
    case ``anc_reset = False``.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 1 for q in layout.data_qubits}
    if layout.num_data_qubits % 2 == 0:
        # ensure that the bistring corresponds to the |-> state
        data_init[layout.data_qubits[-1]] = 0

    yield from init_qubits_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=True
    )


gate_to_iterator = {
    "TICK": qec_round_iterator,
    "I": idle_iterator,
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

from collections.abc import Iterator
from itertools import chain

from stim import Circuit

from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors, get_new_stab_dict_from_layout
from .decorators import (
    qubit_init_z,
    qubit_init_x,
    sq_gate,
    tq_gate,
    logical_measurement_z,
    logical_measurement_x,
)


def qubit_coords(model: Model, *layouts: Layout) -> Circuit:
    """Returns a stim circuit that sets up the coordinates of the qubits."""
    coord_dict = {}
    for layout in layouts:
        coord_dict.update(layout.qubit_coords)
    return model.qubit_coords(coord_dict)


@sq_gate
def idle_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuit blocks which in total correspond to a logical idling
    of the given model.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    """
    data_qubits = layout.data_qubits
    qubits = layout.qubits

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    yield model.idle(qubits)
    yield model.tick()


def log_meas(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical measurement
    of the given model. It defines the observables for all the logical
    qubits in the layout.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    anc_qubits = layout.anc_qubits
    if anc_detectors is None:
        anc_detectors = anc_qubits
    if set(anc_detectors) > set(anc_qubits):
        raise ValueError("Some of the given 'anc_qubits' are not ancilla qubits.")

    circuit = sum(
        log_meas_iterator(model=model, layout=layout, rot_basis=rot_basis),
        start=Circuit(),
    )

    # detectors and logical observables
    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    anc_support = layout.get_support(stabs)
    detectors_stim = detectors.build_from_data(
        model.meas_target,
        anc_support,
        anc_reset,
        reconstructable_stabs=stabs,
        anc_qubits=anc_detectors,
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    for logical_qubit in layout.logical_qubits:
        log_data_qubits = layout.logical_param(log_op, logical_qubit)
        targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    # deactivate detectors
    detectors.deactivate_detectors(layout.anc_qubits)

    return circuit


def log_meas_iterator(
    model: Model,
    layout: Layout,
    rot_basis: bool = False,
) -> Iterator[Circuit]:
    """
    Yields stim circuit blocks which in total correspond to a logical measurement
    of the given model without the definition of the detectors and observables.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits

    if rot_basis:
        yield model.hadamard(data_qubits) + model.idle(anc_qubits)
        yield model.tick()

    yield model.measure(data_qubits) + model.idle_meas(anc_qubits)
    yield model.tick()


@logical_measurement_z
def log_meas_z_iterator(model: Model, layout: Layout):
    """
    Yields stim circuit blocks which in total correspond to a logical Z measurement
    of the given model without the definition of the detectors and observables.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    """
    yield from log_meas_iterator(model=model, layout=layout, rot_basis=False)


@logical_measurement_x
def log_meas_x_iterator(model: Model, layout: Layout):
    """
    Yields stim circuit blocks which in total correspond to a logical X measurement
    of the given model without the definition of the detectors and observables.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    """
    yield from log_meas_iterator(model=model, layout=layout, rot_basis=True)


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
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    # activate detectors
    # the order of activating the detectors or applying the circuit
    # does not matter because this will be done in a layer of logical operations,
    # so no QEC cycles are run simultaneously
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
) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization
    of the given model.
    By default, the logical initialization is in the Z basis.
    If rot_basis, the logical initialization is in the X basis.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)

    yield model.reset(data_qubits) + model.idle(anc_qubits)
    yield model.tick()

    init_circ = Circuit()
    exc_qubits = set([q for q, s in data_init.items() if s and (q in data_qubits)])
    if exc_qubits:
        init_circ += model.x_gate(exc_qubits)

    idle_qubits = qubits - exc_qubits
    yield init_circ + model.idle(idle_qubits)
    yield model.tick()

    if rot_basis:
        yield model.hadamard(data_qubits) + model.idle(anc_qubits)
        yield model.tick()


@qubit_init_z
def init_qubits_z0_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |0>
    state of the given model.

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
def init_qubits_z1_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |1>
    state of the given model.

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
def init_qubits_x0_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |+>
    state of the given model.

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
def init_qubits_x1_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |->
    state of the given model.

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


def log_x(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical X gate
    of the given model.
    """
    # the stabilizer generators do not change when applying a logical X gate
    return sum(log_x_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_x_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical X gate
    of the given model.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits
    qubits = anc_qubits + data_qubits

    log_qubit_label = layout.logical_qubits[0]
    log_x_qubits = layout.logical_param("log_x", log_qubit_label)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    idle_qubits = set(qubits) - set(log_x_qubits)
    yield model.x_gate(log_x_qubits) + model.idle(idle_qubits)
    yield model.tick()


def log_z(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical Z gate
    of the given model.
    """
    # the stabilizer generators do not change when applying a logical Z gate
    return sum(log_z_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_z_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical Z gate
    of the given model.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits
    qubits = anc_qubits + data_qubits

    log_qubit_label = layout.logical_qubits[0]
    log_z_qubits = layout.logical_param("log_z", log_qubit_label)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    idle_qubits = set(qubits) - set(log_z_qubits)
    yield model.z_gate(log_z_qubits) + model.idle(idle_qubits)
    yield model.tick()


def log_fold_trans_s(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """Returns the stim circuit corresponding to a transversal logical S gate
    implemented following:

    https://quantum-journal.org/papers/q-2024-04-08-1310/

    and

    https://doi.org/10.1088/1367-2630/17/8/083026
    """
    # update the stabilizer generators
    gate_label = f"log_fold_trans_s_{layout.logical_qubits[0]}"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(log_fold_trans_s_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_fold_trans_s_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """Yields the stim circuits corresponding to a transversal logical S gate
    implemented following:

    https://quantum-journal.org/papers/q-2024-04-08-1310/

    and

    https://doi.org/10.1088/1367-2630/17/8/083026
    """
    if layout.code not in ["rotated_surface_code", "unrotated_surface_code"]:
        raise TypeError(
            "The given layout is not a rotated/unrotated surface code, "
            f"but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    gate_label = f"log_fold_trans_s_{layout.logical_qubits[0]}"

    cz_pairs = set()
    qubits_s_gate = set()
    qubits_s_dag_gate = set()
    for data_qubit in data_qubits:
        trans_s = layout.param(gate_label, data_qubit)
        if trans_s is None:
            raise ValueError(
                "The layout does not have the information to run a "
                f"transversal S gate on qubit {data_qubit}. "
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
    """Returns the stim circuit corresponding to a transversal logical H gate
    implemented following the circuit show in:

    https://arxiv.org/pdf/2406.17653
    """
    # update the stabilizer generators
    gate_label = f"log_fold_trans_h_{layout.logical_qubits[0]}"
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(log_fold_trans_h_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_fold_trans_h_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """Yields the stim circuits corresponding to a fold-transversal logical H gate
    implemented following the circuit show in:

    https://arxiv.org/pdf/2406.17653
    """
    if layout.code not in ["unrotated_surface_code"]:
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}"
        )

    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)
    gate_label = f"log_fold_trans_h_{layout.logical_qubits[0]}"

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


def log_trans_cnot(
    model: Model, layout_c: Layout, layout_t: Layout, detectors: Detectors
) -> Circuit:
    """Returns the stim circuit corresponding to a transversal logical CNOT gate.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout_c
        Code layout for the control of the logical CNOT.
    layout_t
        Code layout for the target of the logical CNOT.
    detectors
        Detector definitions to use.
    """
    # update the stabilizer generators
    gate_label = (
        f"log_trans_cnot_{layout_c.logical_qubits[0]}_{layout_t.logical_qubits[0]}"
    )
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout_c, gate_label)
    new_stabs_2, new_stabs_2_inv = get_new_stab_dict_from_layout(layout_t, gate_label)
    new_stabs.update(new_stabs_2)
    new_stabs_inv.update(new_stabs_2_inv)
    detectors.update(new_stabs, new_stabs_inv)
    return sum(
        log_trans_cnot_iterator(model=model, layout_c=layout_c, layout_t=layout_t),
        start=Circuit(),
    )


@tq_gate
def log_trans_cnot_iterator(
    model: Model, layout_c: Layout, layout_t: Layout
) -> Iterator[Circuit]:
    """Returns the stim circuit corresponding to a transversal logical CNOT gate.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout_c
        Code layout for the control of the logical CNOT.
    layout_t
        Code layout for the target of the logical CNOT.
    detectors
        Detector definitions to use.
    """
    if layout_c.code not in ["rotated_surface_code", "unrotated_surface_code"]:
        raise TypeError(
            "The given layout is not a rotated/unrotated surface code, "
            f"but a {layout_c.code}"
        )
    if layout_t.code not in ["rotated_surface_code", "unrotated_surface_code"]:
        raise TypeError(
            "The given layout is not a rotated/unrotated surface code, "
            f"but a {layout_t.code}"
        )

    data_qubits_c = layout_c.data_qubits
    data_qubits_t = layout_t.data_qubits
    qubits = set(layout_c.qubits + layout_t.qubits)
    gate_label = (
        f"log_trans_cnot_{layout_c.logical_qubits[0]}_{layout_t.logical_qubits[0]}"
    )

    cz_pairs = set()
    qubits_h_gate = set(data_qubits_t)
    for data_qubit in data_qubits_c:
        trans_cnot = layout_c.param(gate_label, data_qubit)
        if trans_cnot is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        cz_pairs.add((data_qubit, trans_cnot["cz"]))

    yield model.incoming_noise(data_qubits_c) + model.incoming_noise(data_qubits_t)
    yield model.tick()

    # CNOT is decomposed into H CZ H
    idle_qubits = qubits - qubits_h_gate
    yield model.hadamard(qubits_h_gate) + model.idle(idle_qubits)
    yield model.tick()

    # long-range CZ gates
    int_qubits = list(chain.from_iterable(cz_pairs))
    idle_qubits = qubits - set(int_qubits)
    yield model.cphase(int_qubits) + model.idle(idle_qubits)
    yield model.tick()

    # CNOT is decomposed into H CZ H
    idle_qubits = qubits - qubits_h_gate
    yield model.hadamard(qubits_h_gate) + model.idle(idle_qubits)
    yield model.tick()


def log_meas_xzzx(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical measurement
    of the given model. It defines the observables for all the logical
    qubits in the layout.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}"
        )

    if anc_detectors is None:
        anc_detectors = layout.anc_qubits
    if set(anc_detectors) > set(layout.anc_qubits):
        raise ValueError("Some of the given 'anc_qubits' are not ancilla qubits.")

    circuit = sum(
        log_meas_xzzx_iterator(model=model, layout=layout, rot_basis=rot_basis),
        start=Circuit(),
    )

    # detectors and logical observables
    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    anc_support = layout.get_support(stabs)
    detectors_stim = detectors.build_from_data(
        model.meas_target,
        anc_support,
        anc_reset,
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


def log_meas_xzzx_iterator(
    model: Model,
    layout: Layout,
    rot_basis: bool = False,
) -> Iterator[Circuit]:
    """
    Yields stim circuit blocks which in total correspond to a logical measurement
    of the given model without the definition of the detectors and observables.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits
    qubits = set(layout.qubits)

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)
    idle_qubits = qubits - rot_qubits

    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()

    yield model.measure(data_qubits) + model.idle_meas(anc_qubits)
    yield model.tick()


@logical_measurement_z
def log_meas_z_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuit blocks which in total correspond to a logical Z measurement
    of the given model without the definition of the detectors and observables.
    """
    yield from log_meas_xzzx_iterator(model=model, layout=layout, rot_basis=False)


@logical_measurement_x
def log_meas_x_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuit blocks which in total correspond to a logical X measurement
    of the given model without the definition of the detectors and observables.
    """
    yield from log_meas_xzzx_iterator(model=model, layout=layout, rot_basis=True)


def log_x_xzzx(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical X gate
    of the given model.
    """
    # the stabilizer generators do not change when applying a logical X gate
    return sum(log_x_xzzx_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_x_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical X gate
    of the given model.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits

    log_qubit_label = layout.logical_qubits[0]
    log_x_qubits = layout.logical_param("log_x", log_qubit_label)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # apply log X
    rot_qubits = []
    stab_qubits = layout.get_qubits(role="anc", stab_type="z_type")
    for direction in ("north_west", "south_east"):
        rot_qubits += layout.get_neighbors(stab_qubits, direction=direction)
    pauli_z = set(d for d in log_x_qubits if d in rot_qubits)
    pauli_x = set(log_x_qubits) - pauli_z

    yield model.x_gate(pauli_x) + model.z_gate(pauli_z) + model.idle(anc_qubits)
    yield model.tick()


def log_z_xzzx(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical Z gate
    of the given model.
    """
    # the stabilizer generators do not change when applying a logical Z gate
    return sum(log_z_xzzx_iterator(model=model, layout=layout), start=Circuit())


@sq_gate
def log_z_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical Z gate
    of the given model.
    """
    anc_qubits = layout.anc_qubits
    data_qubits = layout.data_qubits

    log_qubit_label = layout.logical_qubits[0]
    log_z_qubits = layout.logical_param("log_z", log_qubit_label)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    # apply log Z
    rot_qubits = []
    stab_qubits = layout.get_qubits(role="anc", stab_type="z_type")
    for direction in ("north_west", "south_east"):
        rot_qubits += layout.get_neighbors(stab_qubits, direction=direction)
    pauli_x = set(d for d in log_z_qubits if d in rot_qubits)
    pauli_z = set(log_z_qubits) - pauli_x

    yield model.x_gate(pauli_x) + model.z_gate(pauli_z) + model.idle(anc_qubits)
    yield model.tick()


def init_qubits_xzzx(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    data_init: dict[str, int],
    rot_basis: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical initialization
    of the given model.
    By default, the logical initialization is in the Z basis.
    If rot_basis, the logical initialization is in the X basis.
    """
    # activate detectors
    # the order of activating the detectors or applying the circuit
    # does not matter because this will be done in a layer of logical operations,
    # so no QEC cycles are run simultaneously
    anc_qubits = layout.anc_qubits
    stab_type = "z_type" if rot_basis else "x_type"
    gauge_dets = layout.get_qubits(role="anc", stab_type=stab_type)
    detectors.activate_detectors(anc_qubits, gauge_dets=gauge_dets)
    return sum(
        init_qubits_xzzx_iterator(
            model=model, layout=layout, data_init=data_init, rot_basis=rot_basis
        ),
        start=Circuit(),
    )


def init_qubits_xzzx_iterator(
    model: Model,
    layout: Layout,
    data_init: dict[str, int],
    rot_basis: bool = False,
) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}"
        )

    qubits = set(layout.qubits)

    yield model.reset(layout.data_qubits) + model.idle(layout.anc_qubits)
    yield model.tick()

    init_circ = Circuit()
    exc_qubits = set(
        [q for q, s in data_init.items() if s and (q in layout.data_qubits)]
    )
    if exc_qubits:
        init_circ += model.x_gate(exc_qubits)

    idle_qubits = qubits - exc_qubits
    yield init_circ + model.idle(idle_qubits)
    yield model.tick()

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    idle_qubits = qubits - rot_qubits
    yield model.hadamard(rot_qubits) + model.idle(idle_qubits)
    yield model.tick()


@qubit_init_z
def init_qubits_z0_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |0>
    state of the given model.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 0 for q in layout.data_qubits}
    yield from init_qubits_xzzx_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=False
    )


@qubit_init_z
def init_qubits_z1_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |1>
    state of the given model.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 1 for q in layout.data_qubits}
    if layout.num_data_qubits % 2 == 0:
        # ensure that the bistring corresponds to the |1> state
        data_init[layout.data_qubits[-1]] = 0

    yield from init_qubits_xzzx_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=False
    )


@qubit_init_x
def init_qubits_x0_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |+>
    state of the given model.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 0 for q in layout.data_qubits}
    yield from init_qubits_xzzx_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=True
    )


@qubit_init_x
def init_qubits_x1_xzzx_iterator(model: Model, layout: Layout) -> Iterator[Circuit]:
    """
    Yields stim circuits corresponding to a logical initialization in the |->
    state of the given model.

    Notes
    -----
    The ``data_init`` bitstring used for the initialization is not important
    when doing stabilizer simulation.
    """
    data_init = {q: 1 for q in layout.data_qubits}
    if layout.num_data_qubits % 2 == 0:
        # ensure that the bistring corresponds to the |-> state
        data_init[layout.data_qubits[-1]] = 0

    yield from init_qubits_xzzx_iterator(
        model=model, layout=layout, data_init=data_init, rot_basis=True
    )

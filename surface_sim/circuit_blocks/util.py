from collections.abc import Iterator
from itertools import chain

from stim import Circuit

from ..layouts import Layout
from ..models import Model
from ..detectors import (
    Detectors,
    get_new_stab_dict_from_layout,
    get_support_from_adj_matrix,
)


def qubit_coords(model: Model, layout: Layout) -> Circuit:
    """Returns a stim circuit that sets up the coordinates
    of the qubits.
    """
    coord_dict = layout.qubit_coords()
    circuit = Circuit()

    circuit += model.qubit_coords(coord_dict)

    return circuit


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
    of the given model.

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
    anc_qubits = layout.get_qubits(role="anc")
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
    anc_support = get_support_from_adj_matrix(layout.adjacency_matrix(), stabs)
    detectors_stim = detectors.build_from_data(
        model.meas_target,
        anc_support,
        anc_reset,
        reconstructable_stabs=stabs,
        anc_qubits=anc_detectors,
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    log_qubits_support = getattr(layout, log_op)
    log_qubit_label = layout.get_logical_qubits()[0]
    log_data_qubits = log_qubits_support[log_qubit_label]
    targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    # deactivate detectors
    detectors.deactivate_detectors(layout.get_qubits(role="anc"))

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
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    if rot_basis:
        yield model.hadamard(data_qubits)
        yield model.idle(anc_qubits)
        yield model.tick()

    yield model.measure(data_qubits)
    yield model.idle(anc_qubits)
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
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()
    circuit += model.reset(qubits)
    circuit += model.tick()

    exc_qubits = set([q for q, s in data_init.items() if s])
    if exc_qubits:
        circuit += model.x_gate(exc_qubits)

    idle_qubits = qubits - exc_qubits
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    if rot_basis:
        circuit += model.hadamard(data_qubits)
        circuit += model.idle(anc_qubits)
        circuit += model.tick()

    # activate detectors
    detectors.activate_detectors(anc_qubits)

    return circuit


def log_x(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical X gate
    of the given model.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = anc_qubits + data_qubits

    log_qubit_label = layout.get_logical_qubits()[0]
    log_x_qubits = layout.log_x[log_qubit_label]

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    idle_qubits = set(qubits) - set(log_x_qubits)
    circuit += model.x_gate(log_x_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # the stabilizer generators do not change when applying a logical X gate

    return circuit


def log_z(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical Z gate
    of the given model.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = anc_qubits + data_qubits

    log_qubit_label = layout.get_logical_qubits()[0]
    log_z_qubits = layout.log_z[log_qubit_label]

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    idle_qubits = set(qubits) - set(log_z_qubits)
    circuit += model.z_gate(log_z_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # the stabilizer generators do not change when applying a logical Z gate

    return circuit


def log_trans_s(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """Returns the stim circuit corresponding to a transversal logical S gate
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

    data_qubits = layout.get_qubits(role="data")
    qubits = set(layout.get_qubits())
    gate_label = f"trans-s_{layout.get_logical_qubits()[0]}"

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

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    # S, S_DAG gates
    idle_qubits = qubits - qubits_s_gate - qubits_s_dag_gate
    circuit += model.s_gate(qubits_s_gate)
    circuit += model.s_dag_gate(qubits_s_dag_gate)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # long-range CZ gates
    int_qubits = list(chain.from_iterable(cz_pairs))
    idle_qubits = qubits - set(int_qubits)
    circuit += model.cphase(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # update the stabilizer generators
    new_stabs = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update_from_dict(new_stabs)

    return circuit


def log_trans_h(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """Returns the stim circuit corresponding to a transversal logical H gate
    implemented following the circuit show in:

    https://arxiv.org/pdf/2406.17653
    """
    if layout.code not in ["unrotated_surface_code"]:
        raise TypeError(
            f"The given layout is not an unrotated surface code, but a {layout.code}"
        )

    data_qubits = layout.get_qubits(role="data")
    qubits = set(layout.get_qubits())
    gate_label = f"trans-h_{layout.get_logical_qubits()[0]}"

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

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    # H gates
    idle_qubits = qubits - qubits_h_gate
    circuit += model.hadamard(qubits_h_gate)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # long-range SWAP gates
    int_qubits = list(chain.from_iterable(swap_pairs))
    idle_qubits = qubits - set(int_qubits)
    circuit += model.swap(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # update the stabilizer generators
    new_stabs = get_new_stab_dict_from_layout(layout, gate_label)
    detectors.update_from_dict(new_stabs)

    return circuit


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

    data_qubits_c = layout_c.get_qubits(role="data")
    data_qubits_t = layout_t.get_qubits(role="data")
    qubits = set(layout_c.get_qubits() + layout_t.get_qubits())
    gate_label = f"trans-cnot_{layout_c.get_logical_qubits()[0]}_{layout_t.get_logical_qubits()[0]}"

    cnot_pairs = set()
    for data_qubit in data_qubits_c:
        trans_cnot = layout_c.param(gate_label, data_qubit)
        if trans_cnot is None:
            raise ValueError(
                "The layout does not have the information to run "
                f"{gate_label} gate on qubit {data_qubit}. "
                "Use the 'log_gates' module to set it up."
            )
        cnot_pairs.add((data_qubit, trans_cnot["cnot"]))

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits_c)
    circuit += model.incoming_noise(data_qubits_t)
    circuit += model.tick()

    # long-range CNOT gates
    int_qubits = list(chain.from_iterable(cnot_pairs))
    idle_qubits = qubits - set(int_qubits)
    circuit += model.cnot(int_qubits)
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # update the stabilizer generators
    new_stabs = get_new_stab_dict_from_layout(layout_c, gate_label)
    new_stabs.update(get_new_stab_dict_from_layout(layout_t, gate_label))
    detectors.update_from_dict(new_stabs)

    return circuit


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
    of the given model.

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
        anc_detectors = layout.get_qubits(role="anc")
    if set(anc_detectors) > set(layout.get_qubits(role="anc")):
        raise ValueError("Some of the given 'anc_qubits' are not ancilla qubits.")

    circuit = sum(
        log_meas_xzzx_iterator(model=model, layout=layout, rot_basis=rot_basis),
        start=Circuit(),
    )

    # detectors and logical observables
    stab_type = "x_type" if rot_basis else "z_type"
    stabs = layout.get_qubits(role="anc", stab_type=stab_type)
    anc_support = get_support_from_adj_matrix(layout.adjacency_matrix(), stabs)
    detectors_stim = detectors.build_from_data(
        model.meas_target,
        anc_support,
        anc_reset,
        reconstructable_stabs=stabs,
        anc_qubits=anc_detectors,
    )
    circuit += detectors_stim

    log_op = "log_x" if rot_basis else "log_z"
    log_qubits_support = getattr(layout, log_op)
    log_qubit_label = layout.get_logical_qubits()[0]
    log_data_qubits = log_qubits_support[log_qubit_label]
    targets = [model.meas_target(qubit, -1) for qubit in log_data_qubits]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    detectors.deactivate_detectors(layout.get_qubits(role="anc"))

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
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    yield model.incoming_noise(data_qubits)
    yield model.tick()

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)
    idle_qubits = qubits - rot_qubits

    yield model.hadamard(rot_qubits)
    yield model.idle(idle_qubits)
    yield model.tick()

    yield model.measure(data_qubits)
    yield model.idle(anc_qubits)
    yield model.tick()


def log_x_xzzx(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical X gate
    of the given model.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    log_qubit_label = layout.get_logical_qubits()[0]
    log_x_qubits = layout.log_x[log_qubit_label]

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    # apply log X
    rot_qubits = []
    stab_qubits = layout.get_qubits(role="anc", stab_type="z_type")
    for direction in ("north_west", "south_east"):
        rot_qubits += layout.get_neighbors(stab_qubits, direction=direction)
    pauli_z = set(d for d in log_x_qubits if d in rot_qubits)
    pauli_x = set(log_x_qubits) - pauli_z

    circuit += model.x_gate(pauli_x)
    circuit += model.z_gate(pauli_z)
    circuit += model.idle(anc_qubits)
    circuit += model.tick()

    # the stabilizer generators do not change when applying a logical X gate

    return circuit


def log_z_xzzx(model: Model, layout: Layout, detectors: Detectors) -> Circuit:
    """
    Returns stim circuit corresponding to a logical Z gate
    of the given model.
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    log_qubit_label = layout.get_logical_qubits()[0]
    log_z_qubits = layout.log_z[log_qubit_label]

    circuit = Circuit()

    circuit += model.incoming_noise(data_qubits)
    circuit += model.tick()

    # apply log Z
    rot_qubits = []
    stab_qubits = layout.get_qubits(role="anc", stab_type="z_type")
    for direction in ("north_west", "south_east"):
        rot_qubits += layout.get_neighbors(stab_qubits, direction=direction)
    pauli_x = set(d for d in log_z_qubits if d in rot_qubits)
    pauli_z = set(log_z_qubits) - pauli_x

    circuit += model.x_gate(pauli_x)
    circuit += model.z_gate(pauli_z)
    circuit += model.idle(anc_qubits)
    circuit += model.tick()

    # the stabilizer generators do not change when applying a logical Z gate

    return circuit


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
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    if layout.code != "rotated_surface_code":
        raise TypeError(
            f"The given layout is not a rotated surface code, but a {layout.code}"
        )

    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()
    circuit += model.reset(qubits)
    circuit += model.tick()

    exc_qubits = set([q for q, s in data_init.items() if s])
    if exc_qubits:
        circuit += model.x_gate(exc_qubits)

    idle_qubits = qubits - exc_qubits
    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    circuit += model.hadamard(rot_qubits)

    idle_qubits = qubits - rot_qubits

    circuit += model.idle(idle_qubits)
    circuit += model.tick()

    # activate detectors
    detectors.activate_detectors(anc_qubits)

    return circuit

from ..layouts.layout import Layout


def set_x(layout: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the Pauli X
    gate for the unrotated surface code.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    """
    if len(layout.get_logical_qubits()) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.get_logical_qubits())}."
        )

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    log_qubit_label = layout.get_logical_qubits()[0]
    gate_label = f"log_x_{log_qubit_label}"

    x_gates = {q: "I" for q in data_qubits}
    for q in layout.log_x[log_qubit_label]:
        x_gates[q] = "X"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"local": x_gates[qubit]})

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(gate_label, anc_qubit, {"new_stab_gen": [anc_qubit]})

    return


def set_z(layout: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the Pauli Z
    gate for the unrotated surface code.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    """
    if len(layout.get_logical_qubits()) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.get_logical_qubits())}."
        )

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    log_qubit_label = layout.get_logical_qubits()[0]
    gate_label = f"log_z_{log_qubit_label}"

    z_gates = {q: "I" for q in data_qubits}
    for q in layout.log_z[log_qubit_label]:
        z_gates[q] = "Z"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"local": z_gates[qubit]})

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(gate_label, anc_qubit, {"new_stab_gen": [anc_qubit]})

    return


def set_idle(layout: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the Pauli I
    gate for the unrotated surface code.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    """
    if len(layout.get_logical_qubits()) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.get_logical_qubits())}."
        )

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    log_qubit_label = layout.get_logical_qubits()[0]
    gate_label = f"idle_{log_qubit_label}"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"local": "I"})

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(gate_label, anc_qubit, {"new_stab_gen": [anc_qubit]})

    return

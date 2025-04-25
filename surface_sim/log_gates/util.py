from ..layouts.layout import Layout
from ..layouts.operations import check_overlap_layouts


def set_x(layout: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the Pauli X
    gate for the unrotated surface code.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    """
    if len(layout.logical_qubits) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.logical_qubits)}."
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    log_qubit_label = layout.logical_qubits[0]
    gate_label = f"log_x_{log_qubit_label}"

    x_gates = {q: "I" for q in data_qubits}
    for q in layout.logical_param("log_x", log_qubit_label):
        x_gates[q] = "X"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"local": x_gates[qubit]})

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(
            gate_label,
            anc_qubit,
            {"new_stab_gen": [anc_qubit], "new_stab_gen_inv": [anc_qubit]},
        )

    return


def set_z(layout: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the Pauli Z
    gate for the unrotated surface code.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    """
    if len(layout.logical_qubits) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.logical_qubits)}."
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    log_qubit_label = layout.logical_qubits[0]
    gate_label = f"log_z_{log_qubit_label}"

    z_gates = {q: "I" for q in data_qubits}
    for q in layout.logical_param("log_z", log_qubit_label):
        z_gates[q] = "Z"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"local": z_gates[qubit]})

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(
            gate_label,
            anc_qubit,
            {"new_stab_gen": [anc_qubit], "new_stab_gen_inv": [anc_qubit]},
        )

    return


def set_idle(layout: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the Pauli I
    gate for the unrotated surface code.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    """
    if len(layout.logical_qubits) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.logical_qubits)}."
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    log_qubit_label = layout.logical_qubits[0]
    gate_label = f"idle_{log_qubit_label}"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"local": "I"})

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(
            gate_label,
            anc_qubit,
            {"new_stab_gen": [anc_qubit], "new_stab_gen_inv": [anc_qubit]},
        )

    return


def set_trans_cnot(layout_c: Layout, layout_t: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the
    transversal CNOT gate for the unrotated surface code.

    Parameters
    ----------
    layout_c
        The layout for the control of the CNOT for which to add the attributes.
    layout_t
        The layout for the target of the CNOT for which to add the attributes.
    """
    if (layout_c.code not in ["unrotated_surface_code", "rotated_surface_code"]) or (
        layout_t.code not in ["unrotated_surface_code", "rotated_surface_code"]
    ):
        raise ValueError(
            "This function is for unrotated and rotated surface codes, "
            f"but layouts for {layout_t.code} and {layout_c.code} were given."
        )
    if (layout_c.distance_x != layout_t.distance_x) or (
        layout_c.distance_z != layout_t.distance_z
    ):
        raise ValueError("This function requires two surface codes of the same size.")
    check_overlap_layouts(layout_c, layout_t)

    gate_label = (
        f"log_trans_cnot_{layout_c.logical_qubits[0]}_{layout_t.logical_qubits[0]}"
    )

    qubit_coords_c = layout_c.qubit_coords
    qubit_coords_t = layout_t.qubit_coords
    bottom_left_qubit_c = sorted(
        qubit_coords_c.items(), key=lambda x: 999_999_999 * x[1][0] + x[1][1]
    )
    bottom_left_qubit_t = sorted(
        qubit_coords_t.items(), key=lambda x: 999_999_999 * x[1][0] + x[1][1]
    )
    mapping_t_to_c = {}
    mapping_c_to_t = {}
    for (qc, _), (qt, _) in zip(bottom_left_qubit_c, bottom_left_qubit_t):
        mapping_t_to_c[qt] = qc
        mapping_c_to_t[qc] = qt

    # Store the logical information for the data qubits
    data_qubits_c = set(layout_c.data_qubits)
    data_qubits_t = set(layout_t.data_qubits)
    for qubit in data_qubits_c:
        layout_c.set_param(
            gate_label, qubit, {"cz": mapping_c_to_t[qubit], "local": "I"}
        )
    for qubit in data_qubits_t:
        layout_t.set_param(
            gate_label, qubit, {"cz": mapping_t_to_c[qubit], "local": "H"}
        )

    # Compute the new stabilizer generators based on the CNOT connections
    anc_to_new_stab = {}
    for anc in layout_c.get_qubits(role="anc", stab_type="z_type"):
        anc_to_new_stab[anc] = [anc]
    for anc in layout_c.get_qubits(role="anc", stab_type="x_type"):
        anc_to_new_stab[anc] = [anc, mapping_c_to_t[anc]]
    for anc in layout_t.get_qubits(role="anc", stab_type="z_type"):
        anc_to_new_stab[anc] = [anc, mapping_t_to_c[anc]]
    for anc in layout_t.get_qubits(role="anc", stab_type="x_type"):
        anc_to_new_stab[anc] = [anc]

    # Store new stabilizer generators to the ancilla qubits
    # CNOT^\dagger = CNOT
    for anc in layout_c.anc_qubits:
        layout_c.set_param(
            gate_label,
            anc,
            {
                "new_stab_gen": anc_to_new_stab[anc],
                "new_stab_gen_inv": anc_to_new_stab[anc],
            },
        )
    for anc in layout_t.anc_qubits:
        layout_t.set_param(
            gate_label,
            anc,
            {
                "new_stab_gen": anc_to_new_stab[anc],
                "new_stab_gen_inv": anc_to_new_stab[anc],
            },
        )

    return

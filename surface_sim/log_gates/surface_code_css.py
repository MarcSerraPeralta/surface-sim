from copy import deepcopy

import numpy as np

from ..layouts import Layout


def set_trans_s(layout: Layout, data_qubit: str) -> None:
    """Adds the required attributes (in place) for the layout to run the transversal S
    gate for the rotated surface code.

    This implementation assumes that the qubits are placed in a square 2D grid.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    data_qubit
        The data qubit in a corner through which the folding of the surface
        code runs.
    """
    if layout.code != "rotated_surface_code":
        raise ValueError(
            "This function is for rotated surface codes, "
            f"but a layout for the code {layout.code} was given."
        )
    if layout.distance_z != layout.distance_x + 1:
        raise ValueError("The transversal S gate requires d_z = d_x + 1.")
    if data_qubit not in layout.get_qubits(role="data"):
        raise ValueError(f"{data_qubit} is not a data qubit from the given layout.")
    if set(map(len, layout.get_coords(layout.get_qubits()))) != {2}:
        raise ValueError("The qubit coordinates must be 2D.")

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    stab_x = layout.get_qubits(role="anc", stab_type="x_type")
    stab_z = layout.get_qubits(role="anc", stab_type="z_type")

    # get the jump coordinates
    neighbors = layout.param("neighbors", data_qubit)
    dir_x, anc_qubit_x = [(d, q) for d, q in neighbors.items() if q in stab_x][0]
    dir_z, anc_qubit_z = [(d, q) for d, q in neighbors.items() if q in stab_z][0]
    data_qubit_h = layout.get_neighbors(anc_qubit_x, direction=dir_z)[0]

    jump_h = np.array(layout.param("coords", data_qubit_h)) - np.array(
        layout.param("coords", data_qubit)
    )
    jump_v = np.array([jump_h[1], -jump_h[0]])  # perpendicular vector
    data_qubit_coords = np.array(layout.param("coords", data_qubit))

    # get the CZs from the data qubit positions
    coords_to_label_dict = {
        tuple(attr["coords"]): node for node, attr in layout.graph.nodes.items()
    }

    def coords_to_label(c):
        c = tuple(c)
        if c not in coords_to_label_dict:
            return None
        else:
            return coords_to_label_dict[c]

    top_column = deepcopy(data_qubit_coords)
    curr_level = 0
    cz_gates = {}
    while True:
        if coords_to_label(top_column) is None:
            break

        coords1 = top_column + curr_level * jump_v
        label1 = coords_to_label(coords1)

        if label1 is None:
            top_column += jump_v + jump_h
            curr_level = 0
            continue

        coords2 = top_column + (curr_level + 1) * jump_h
        label2 = coords_to_label(coords2)
        cz_gates[label2] = label1
        cz_gates[label1] = label2

        curr_level += 1

    # get S gates from the data qubit positions
    s_gates = {q: "I" for q in data_qubits}
    s_gates[data_qubit] = "S"
    coords = deepcopy(data_qubit_coords) + jump_h
    while True:
        label = coords_to_label(coords)

        if label is None:
            break

        s_gates[label] = "S"
        coords += jump_h + jump_v

    label = coords_to_label(coords - jump_h - jump_v)
    s_gates[label] = "S_DAG"

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(
            "trans_s", qubit, {"cz": cz_gates[qubit], "local": s_gates[qubit]}
        )

    # Compute the new stabilizer generators based on the CZs connections
    # as 'set' is not hashable, I use tuple(sorted(...))...
    anc_to_xstab = {
        anc_qubit: tuple(sorted(layout.get_neighbors([anc_qubit])))
        for anc_qubit in stab_x
    }
    zstab_to_anc = {
        tuple(sorted(layout.get_neighbors([anc_qubit]))): anc_qubit
        for anc_qubit in stab_z
    }

    anc_to_new_stab = {}
    for anc_x, stab in anc_to_xstab.items():
        if anc_x == anc_qubit_x:
            anc_to_new_stab[anc_x] = [anc_x]
            continue

        z_stab = set()
        for d in stab:
            if s_gates[d] == "I":
                z_stab.symmetric_difference_update([cz_gates[d]])
            else:
                z_stab.symmetric_difference_update([d, cz_gates[d]])

        z_stab = tuple(sorted(z_stab))
        anc_z = zstab_to_anc[z_stab]
        anc_to_new_stab[anc_x] = [anc_x, anc_z]
        anc_to_new_stab[anc_z] = [anc_z]

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(
            "trans_s", anc_qubit, {"new_stab_gen": anc_to_new_stab[anc_qubit]}
        )

    return

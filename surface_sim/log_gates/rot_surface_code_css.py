from copy import deepcopy

import numpy as np
import networkx as nx

from ..layouts import Layout, check_overlap_layouts


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

    Notes
    -----
    The circuit implementation follows from https://doi.org/10.22331/q-2024-04-08-1310.
    The information about the logical transversal S gate is stored in the layout
    as the parameter ``"trans-s_{log_qubit_label}"`` for each of the qubits,
    where for the case of data qubits it is the information about which gates
    to perform and for the case of the ancilla qubits it corresponds to
    how the stabilizers generators are transformed.
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
    if len(layout.get_logical_qubits()) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.get_logical_qubits())}."
        )

    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    stab_x = layout.get_qubits(role="anc", stab_type="x_type")
    stab_z = layout.get_qubits(role="anc", stab_type="z_type")
    gate_label = f"trans-s_{layout.get_logical_qubits()[0]}"

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
            gate_label, qubit, {"cz": cz_gates[qubit], "local": s_gates[qubit]}
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
            gate_label, anc_qubit, {"new_stab_gen": anc_to_new_stab[anc_qubit]}
        )

    return


def set_trans_cnot(layout_c: Layout, layout_t: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the
    transversal CNOT gate for the rotated surface code.

    Parameters
    ----------
    layout_c
        The layout for the control of the CNOT for which to add the attributes.
    layout_t
        The layout for the target of the CNOT for which to add the attributes.
    """
    if (layout_c.code != "rotated_surface_code") or (
        layout_t.code != "rotated_surface_code"
    ):
        raise ValueError(
            "This function is for rotated surface codes, "
            f"but layouts for {layout_t.code} and {layout_c.code} were given."
        )
    if (layout_c.distance_x != layout_t.distance_x) or (
        layout_c.distance_z != layout_t.distance_z
    ):
        raise ValueError("This function requires two surface codes of the same size.")
    check_overlap_layouts(layout_c, layout_t)

    gate_label = f"trans-cnot_{layout_c.get_logical_qubits()[0]}_{layout_t.get_logical_qubits()[0]}"

    # Obtain the mapping of qubits of one layout to qubits of the other layout
    gm = nx.isomorphism.DiGraphMatcher(layout_c.graph, layout_t.graph)
    if not gm.is_isomorphic:
        raise ValueError(
            "The two given layouts for the t-CNOT are not isomorphic,"
            "meaning that they don't have the same Tanner graph."
        )

    # https://stackoverflow.com/questions/60820193/networkx-digraphmatcher-returns-no-results-on-directed-graphs
    # The solution to the problem described above is to use 'subgraph_monomorphisms_iter'.
    # We find 4 monomorphisms as the graphs are equal up to 90 degree rotations
    monomorphisms = list(gm.subgraph_monomorphisms_iter())
    if len(monomorphisms) == 0:
        raise ValueError(
            "No monomorphism for the layouts has been found. "
            "Check that the layouts have the appropiate graphs."
        )
    # The monomorphism we want is the one in which all the physical CNOTs
    # are parallel between each other.
    mapping_c_to_t = None
    for monomorphism in monomorphisms:
        correct = True

        c, t = next(iter(monomorphism.items()))
        coords1 = np.array(layout_c.get_coords([c])[0])
        coords2 = np.array(layout_t.get_coords([t])[0])
        base_vector = coords2 - coords1
        for c, t in monomorphism.items():
            coords1 = np.array(layout_c.get_coords([c])[0])
            coords2 = np.array(layout_t.get_coords([t])[0])
            vector = coords2 - coords1
            if not np.allclose(vector, base_vector):
                correct = False
                break

        if correct:
            mapping_c_to_t = monomorphism

    if mapping_c_to_t is None:
        raise ValueError("No mapping between layouts could be found.")

    mapping_t_to_c = {v: k for k, v in mapping_c_to_t.items()}

    # Store the logical information for the data qubits
    data_qubits_c = set(layout_c.get_qubits(role="data"))
    data_qubits_t = set(layout_t.get_qubits(role="data"))
    for qubit in data_qubits_c:
        layout_c.set_param(gate_label, qubit, {"cnot": mapping_c_to_t[qubit]})
    for qubit in data_qubits_t:
        layout_t.set_param(gate_label, qubit, {"cnot": mapping_t_to_c[qubit]})

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
    for anc in layout_c.get_qubits(role="anc"):
        layout_c.set_param(gate_label, anc, {"new_stab_gen": anc_to_new_stab[anc]})
    for anc in layout_t.get_qubits(role="anc"):
        layout_t.set_param(gate_label, anc, {"new_stab_gen": anc_to_new_stab[anc]})

    return

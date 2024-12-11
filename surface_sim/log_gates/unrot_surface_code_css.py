import numpy as np
import networkx as nx

from ..layouts import Layout, check_overlap_layouts


def set_trans_s(layout: Layout, data_qubit: str) -> None:
    """Adds the required attributes (in place) for the layout to run the transversal S
    gate for the unrotated surface code.

    This implementation assumes that the qubits are placed in a square 2D grid,
    and the separation between qubits is larger than ``1e-5`` units.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    data_qubit
        The data qubit in a corner through which the folding of the surface
        code runs.

    Notes
    -----
    The circuit implementation follows from https://doi.org/10.1088/1367-2630/17/8/083026
    The circuit is shown in https://arxiv.org/pdf/2406.17653
    The information about the logical transversal S gate is stored in the layout
    as the parameter ``"trans-s_{log_qubit_label}"`` for each of the qubits,
    where for the case of data qubits it is the information about which gates
    to perform and for the case of the ancilla qubits it corresponds to
    how the stabilizers generators are transformed.
    """
    if layout.code != "unrotated_surface_code":
        raise ValueError(
            "This function is for unrotated surface codes, "
            f"but a layout for the code {layout.code} was given."
        )
    if layout.distance_z != layout.distance_x:
        raise ValueError("The transversal S gate requires d_z = d_x.")
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

    # get the reflection function
    neighbors = layout.param("neighbors", data_qubit)
    dir_x, anc_qubit_x = [(d, q) for d, q in neighbors.items() if q in stab_x][0]
    dir_z, anc_qubit_z = [(d, q) for d, q in neighbors.items() if q in stab_z][0]

    data_qubit_diag = layout.get_neighbors(anc_qubit_x, direction=dir_z)[0]
    sym_vector = np.array(layout.param("coords", data_qubit_diag)) - np.array(
        layout.param("coords", data_qubit)
    )
    point = np.array(layout.param("coords", data_qubit))
    fold_reflection = lambda x: reflection(x, point, sym_vector)

    coords_to_label_dict = {}
    for node, attr in layout.graph.nodes.items():
        coords = attr["coords"]
        coords = np.round(coords, decimals=5)  # to avoid numerical issues
        coords_to_label_dict[tuple(coords)] = node

    # get the CZs from the data qubit positions
    cz_gates = {}
    data_qubit_coords = layout.get_coords(data_qubits)
    for data_qubit, coords in zip(data_qubits, data_qubit_coords):
        pair_coords = fold_reflection(coords)
        pair_coords = np.round(pair_coords, decimals=5)
        data_pair = coords_to_label_dict[tuple(pair_coords)]
        cz_gates[data_qubit] = data_pair if data_pair != data_qubit else None

    # get S gates from the data qubit positions
    s_gates = {q: "I" for q in data_qubits}
    for k in range(2 * layout.distance_z - 1):
        coords = point + k * sym_vector
        coords = np.round(coords, decimals=5)
        data_qubit = coords_to_label_dict[tuple(coords)]
        s_gates[data_qubit] = "S" if k % 2 == 0 else "S_DAG"

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
        z_stab = set()
        for d in stab:
            if s_gates[d] == "I":
                z_stab.symmetric_difference_update([cz_gates[d]])
            else:
                z_stab.symmetric_difference_update([d])

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


def reflection(x: np.ndarray, point: np.ndarray, line_vector: np.ndarray) -> np.ndarray:
    """Performs a reflection to ``x`` given the vector and point that define
    the reflection line.
    """
    x = np.array(x)
    theta = -np.arctan(line_vector[1] / line_vector[0])
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    x_rot = rot_matrix @ x
    point = rot_matrix @ point

    x_reflected_rot = np.array([x_rot[0], 2 * point[1] - x_rot[1]])

    theta = -theta
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    x_reflected = rot_matrix @ x_reflected_rot

    return x_reflected


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
    if (layout_c.code != "unrotated_surface_code") or (
        layout_t.code != "unrotated_surface_code"
    ):
        raise ValueError(
            "This function is for unrotated surface codes, "
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


def set_trans_h(layout: Layout, data_qubit: str) -> None:
    """Adds the required attributes (in place) for the layout to run the transversal H
    gate for the unrotated surface code.

    This implementation assumes that the qubits are placed in a square 2D grid,
    and the separation between qubits is larger than ``1e-5`` units.

    Parameters
    ----------
    layout
        The layout in which to add the attributes.
    data_qubit
        The data qubit in a corner through which the folding of the surface
        code runs.

    Notes
    -----
    The circuit is shown in https://arxiv.org/pdf/2406.17653
    The information about the logical transversal H gate is stored in the layout
    as the parameter ``"trans-h_{log_qubit_label}"`` for each of the qubits,
    where for the case of data qubits it is the information about which gates
    to perform and for the case of the ancilla qubits it corresponds to
    how the stabilizers generators are transformed.
    """
    if layout.code != "unrotated_surface_code":
        raise ValueError(
            "This function is for unrotated surface codes, "
            f"but a layout for the code {layout.code} was given."
        )
    if layout.distance_z != layout.distance_x:
        raise ValueError("The transversal H gate requires d_z = d_x.")
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
    gate_label = f"trans-h_{layout.get_logical_qubits()[0]}"

    # get the reflection function
    neighbors = layout.param("neighbors", data_qubit)
    dir_x, anc_qubit_x = [(d, q) for d, q in neighbors.items() if q in stab_x][0]
    dir_z, anc_qubit_z = [(d, q) for d, q in neighbors.items() if q in stab_z][0]

    data_qubit_diag = layout.get_neighbors(anc_qubit_x, direction=dir_z)[0]
    sym_vector = np.array(layout.param("coords", data_qubit_diag)) - np.array(
        layout.param("coords", data_qubit)
    )
    point = np.array(layout.param("coords", data_qubit))
    fold_reflection = lambda x: reflection(x, point, sym_vector)

    coords_to_label_dict = {}
    for node, attr in layout.graph.nodes.items():
        coords = attr["coords"]
        coords = np.round(coords, decimals=5)  # to avoid numerical issues
        coords_to_label_dict[tuple(coords)] = node

    # get the SWAPs from the data qubit positions
    swap_gates = {}
    data_qubit_coords = layout.get_coords(data_qubits)
    for data_qubit, coords in zip(data_qubits, data_qubit_coords):
        pair_coords = fold_reflection(coords)
        pair_coords = np.round(pair_coords, decimals=5)
        data_pair = coords_to_label_dict[tuple(pair_coords)]
        swap_gates[data_qubit] = data_pair if data_pair != data_qubit else None

    # Store logical gate information to the data qubits
    for qubit in data_qubits:
        layout.set_param(gate_label, qubit, {"swap": swap_gates[qubit], "local": "H"})

    # Compute the new stabilizer generators
    anc_to_new_stab = {}
    anc_qubit_coords = layout.get_coords(anc_qubits)
    for anc_qubit, coords in zip(anc_qubits, anc_qubit_coords):
        pair_coords = fold_reflection(coords)
        pair_coords = np.round(pair_coords, decimals=5)
        anc_pair = coords_to_label_dict[tuple(pair_coords)]
        anc_to_new_stab[anc_qubit] = [anc_pair]

    # Store new stabilizer generators to the ancilla qubits
    for anc_qubit in anc_qubits:
        layout.set_param(
            gate_label, anc_qubit, {"new_stab_gen": anc_to_new_stab[anc_qubit]}
        )

    return

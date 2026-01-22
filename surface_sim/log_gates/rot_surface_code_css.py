from copy import deepcopy

import numpy as np

from ..layouts.layout import Layout
from ..layouts.operations import check_overlap_layouts
from .util import set_x, set_z, set_idle, set_trans_cnot

__all__ = [
    "set_x",
    "set_z",
    "set_idle",
    "set_fold_trans_s",
    "set_trans_cnot",
    "set_encoding",
]


def set_fold_trans_s(layout: Layout, data_qubit: str) -> None:
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
    if data_qubit not in layout.data_qubits:
        raise ValueError(f"{data_qubit} is not a data qubit from the given layout.")
    if set(map(len, layout.get_coords(layout.qubits))) != {2}:
        raise ValueError("The qubit coordinates must be 2D.")
    if len(layout.logical_qubits) != 1:
        raise ValueError(
            "The given surface code does not have a logical qubit, "
            f"it has {len(layout.logical_qubits)}."
        )

    data_qubits = layout.data_qubits
    anc_qubits = layout.anc_qubits
    stab_x = layout.get_qubits(role="anc", stab_type="x_type")
    stab_z = layout.get_qubits(role="anc", stab_type="z_type")
    gate_label = f"log_fold_trans_s_{layout.logical_qubits[0]}"

    # get the jump coordinates
    neighbors: dict[str, str] = layout.param("neighbors", data_qubit)
    dir_x, anc_qubit_x = [(d, q) for d, q in neighbors.items() if q in stab_x][0]
    dir_z, anc_qubit_z = [(d, q) for d, q in neighbors.items() if q in stab_z][0]
    data_qubit_h = layout.get_neighbors(anc_qubit_x, direction=dir_z)[0]

    jump_h = np.array(layout.param("coords", data_qubit_h)) - np.array(
        layout.param("coords", data_qubit)
    )
    jump_v = np.array([jump_h[1], -jump_h[0]])  # perpendicular vector
    data_qubit_coords = np.array(layout.param("coords", data_qubit), dtype=float)

    # get the CZs from the data qubit positions
    coords_to_label_dict = {
        tuple(attr["coords"]): node for node, attr in layout.graph.nodes.items()
    }

    def coords_to_label(
        c: tuple[int | float, ...] | np.typing.NDArray[np.float64],
    ) -> None | str:
        c = tuple(c)
        if c not in coords_to_label_dict:
            return None
        else:
            return coords_to_label_dict[c]

    top_column = deepcopy(data_qubit_coords)
    curr_level = 0
    cz_gates: dict[str, str] = {}
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

        z_stab: set[str] = set()
        for d in stab:
            if s_gates[d] == "I":
                z_stab.symmetric_difference_update([cz_gates[d]])
            else:
                z_stab.symmetric_difference_update([d, cz_gates[d]])

        anc_z = zstab_to_anc[tuple(sorted(z_stab))]
        anc_to_new_stab[anc_x] = [anc_x, anc_z]
        anc_to_new_stab[anc_z] = [anc_z]

    # Store new stabilizer generators to the ancilla qubits
    # the stabilizer generator propagation for S_dag is the same for S
    for anc_qubit in anc_qubits:
        layout.set_param(
            gate_label,
            anc_qubit,
            {
                "new_stab_gen": anc_to_new_stab[anc_qubit],
                "new_stab_gen_inv": anc_to_new_stab[anc_qubit],
            },
        )

    return


def set_trans_cnot_mid_cycle_css(layout_c: Layout, layout_t: Layout) -> None:
    """Adds the required attributes (in place) for the layout to run the
    transversal CNOT gate for the rotated surface code in the mid cycle state,
    which corresponds to a CSS unrotated surface code.

    Parameters
    ----------
    layout_c
        The layout for the control of the CNOT for which to add the attributes.
    layout_t
        The layout for the target of the CNOT for which to add the attributes.

    Notes
    -----
    This mid-cycle transversal gate is intended for the QEC cycle the uses CNOTs,
    thus this transversal gate is implemented using CNOTs.
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

    gate_label = f"log_trans_cnot_mid_cycle_css_{layout_c.logical_qubits[0]}_{layout_t.logical_qubits[0]}"

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
    for qubit in layout_c.data_qubits:
        layout_c.set_param(gate_label, qubit, {"cnot": mapping_c_to_t[qubit]})
    for qubit in layout_t.data_qubits:
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
    # CNOT^\dagger = CNOT
    for anc in layout_c.anc_qubits:
        layout_c.set_param(
            gate_label,
            anc,
            {
                "new_stab_gen": anc_to_new_stab[anc],
                "new_stab_gen_inv": anc_to_new_stab[anc],
                "cnot": mapping_c_to_t[anc],
            },
        )
    for anc in layout_t.anc_qubits:
        layout_t.set_param(
            gate_label,
            anc,
            {
                "new_stab_gen": anc_to_new_stab[anc],
                "new_stab_gen_inv": anc_to_new_stab[anc],
                "cnot": mapping_t_to_c[anc],
            },
        )

    return


def set_encoding(layout: Layout) -> None:
    """
    Adds the required attributes (in place) for the layout to run the encoding
    circuits for the rotated surface code.

    This implementation assumes that the qubits are placed in a square 2D grid,
    and that the (spatial) separation between qubits is more than ``1e-6``.

    Parameters
    ----------
    layout
        The (square) layout in which to add the attributes.

    Notes
    -----
    The implementation follows Figure 1 from:

        Claes, Jahan. "Lower-depth local encoding circuits for the surface code."
        arXiv preprint arXiv:2509.09779 (2025).

    The information about the encoding circuit is stored in the layout
    as the parameter ``"encoding_{log_qubit_label}"`` for each of the data qubits.
    """
    if layout.code != "rotated_surface_code":
        raise ValueError(
            "This function is for rotated surface codes, "
            f"but a layout for the code {layout.code} was given."
        )
    if layout.distance_x != layout.distance_z:
        raise ValueError(
            "This function is for square surface codes, "
            f"but d_x={layout.distance_x} and d_z={layout.distance_z} were given."
        )

    gate_label = f"encoding_{layout.logical_qubits[0]}"

    # identify data qubits in the 4 corners of the surface code
    corners: list[str] = []
    for data_qubit in layout.data_qubits:
        if len(layout.get_neighbors([data_qubit])) != 2:
            continue
        corners.append(data_qubit)

    # order the corner qubits following:
    # top left, top right, bottom right, bottom left
    if layout.distance_x % 2 == 1:
        z_corners: list[str] = []
        for data_qubit in corners:
            z_anc = layout.get_neighbors([data_qubit], stab_type="z_type")[0]
            if len(layout.get_neighbors([z_anc])) == 2:
                z_corners.append(data_qubit)
        x_corners = [q for q in corners if q not in z_corners]
        corners = [z_corners[0], x_corners[0], z_corners[1], x_corners[1]]
    else:
        distance: list[float] = []
        coord = np.array(layout.get_coords([corners[0]])[0])
        for q in corners:
            other_coord = np.array(layout.get_coords([q])[0])
            distance.append(((coord - other_coord) ** 2).sum())
        corners = [q for _, q in sorted(zip(distance, corners))]
        corners = [corners[0], corners[1], corners[3], corners[2]]

    # get directions for moving horizontally and vertically
    top_left_coord = np.array(layout.get_coords([corners[0]])[0])
    top_right_coord = np.array(layout.get_coords([corners[1]])[0])
    bottom_right_coord = np.array(layout.get_coords([corners[2]])[0])
    dir_h = (top_right_coord - top_left_coord) / (layout.distance_x - 1)
    dir_v = (bottom_right_coord - top_right_coord) / (layout.distance_x - 1)

    # create mapping from coordinates to data qubit labels
    coord_to_label: dict[tuple[int, ...], str] = {}
    for q, coord in layout.data_coords.items():
        _coord = tuple((np.array(coord) * 1e6).astype(int))
        coord_to_label[_coord] = q

    # extract generalized labels by going around in "outer" rings
    max_rings = (layout.distance_x + 1) // 2
    directions = (dir_h, dir_v, -dir_h, -dir_v)
    glabels: dict[str, tuple[int, int]] = {}
    for r in range(max_rings):
        curr_coord = top_left_coord + r * (dir_h + dir_v)

        _curr_coord = tuple((np.array(curr_coord) * 1e6).astype(int))
        curr_label = coord_to_label[_curr_coord]
        glabels[curr_label] = (max_rings - 1 - r, 0)

        # the most inner ring of odd-distance surface codes contains only
        # a single qubit, not a multiple of 4, thus must be treated differently.
        if (r == max_rings - 1) and (layout.distance_x % 2 == 1):
            continue

        for curr_index in range(1, 4 * (layout.distance_x - 1 - 2 * r)):
            dir = directions[(curr_index - 1) // (layout.distance_x - 1 - 2 * r)]
            curr_coord += dir

            _curr_coord = tuple((np.array(curr_coord) * 1e6).astype(int))
            curr_label = coord_to_label[_curr_coord]
            glabels[curr_label] = (max_rings - 1 - r, curr_index)

    # store generalized labels
    for data_qubit in layout.data_qubits:
        layout.set_param(gate_label, data_qubit, {"label": glabels[data_qubit]})

    return

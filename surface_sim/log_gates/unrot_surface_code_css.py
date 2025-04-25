import numpy as np

from ..layouts.layout import Layout
from .util import set_x, set_z, set_idle, set_trans_cnot

__all__ = [
    "set_x",
    "set_z",
    "set_idle",
    "set_fold_trans_s",
    "set_fold_trans_h",
    "set_trans_cnot",
]


def set_fold_trans_s(layout: Layout, data_qubit: str) -> None:
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
    # the stabilizer propagation for s_dag is the same as for s
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


def set_fold_trans_h(layout: Layout, data_qubit: str) -> None:
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
    gate_label = f"log_fold_trans_h_{layout.logical_qubits[0]}"

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
    # H^\dagger = H
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

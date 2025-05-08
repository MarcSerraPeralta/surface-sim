import numpy as np
import galois

from .layout import Layout


def check_overlap_layout_pair(layout_1: Layout, layout_2: Layout) -> None:
    """Checks if the two given layouts share any qubits for when doing
    parallel logical computation with both of them.

    It checks that there are no shared qubit (1) labels, (2) indices,
    (3) coordinates, and (4) logical qubit labels.

    Parameters
    ----------
    layout_1
        One of the layouts.
    layout_2
        The other layout.
    """
    if not isinstance(layout_1, Layout):
        raise TypeError(f"'layout_1' must be a Layout, but {type(layout_1)} was given.")
    if not isinstance(layout_2, Layout):
        raise TypeError(f"'layout_2' must be a Layout, but {type(layout_2)} was given.")

    qubits_1 = set(layout_1.qubits)
    qubits_2 = set(layout_2.qubits)
    if qubits_1.intersection(qubits_2) != set():
        raise ValueError("The layouts have qubits with the same label.")

    inds_1 = set(layout_1.get_inds(qubits_1))
    inds_2 = set(layout_2.get_inds(qubits_2))
    if inds_1.intersection(inds_2) != set():
        raise ValueError("The layouts have qubits with the same indices.")

    coords_1 = set(map(tuple, layout_1.get_coords(qubits_1)))
    coords_2 = set(map(tuple, layout_2.get_coords(qubits_2)))
    if coords_1.intersection(coords_2) != set():
        raise ValueError("The layouts have qubits with the same coordinates.")

    log_qubits_1 = set(layout_1.logical_qubits)
    log_qubits_2 = set(layout_2.logical_qubits)
    if log_qubits_1.intersection(log_qubits_2) != set():
        raise ValueError("The layouts have logical qubits with the same label.")

    return


def check_overlap_layouts(*layouts: Layout) -> None:
    """Checks if the given layouts share any qubits for when doing
    parallel logical computation with them.

    It checks that there are no shared qubit (1) labels, (2) indices,
    (3) coordinates, and (4) logical qubit labels.

    Parameters
    ----------
    *layouts
        Layouts.
    """
    if len(layouts) == 1:
        return

    for k, layout in enumerate(layouts):
        for other_layout in layouts[k + 1 :]:
            check_overlap_layout_pair(layout, other_layout)

    return


def check_code_definition(layout: Layout) -> None:
    """Checks if the QEC code defined in the layout satisfies the following properties:

    1. stabilisers commute with each other
    2. logical Paulis commute with all stabilisers
    3. logical Xi and Zi anticommute
    4. logical Paulis are independent (i.e. no product of them is in the stabilizer group)
    """
    if not isinstance(layout, Layout):
        raise TypeError(f"'layout' must be a Layout, but {type(layout)} was given.")

    # build parity and logical matrices
    x_anc = layout.get_qubits(role="anc", stab_type="x_type")
    z_anc = layout.get_qubits(role="anc", stab_type="z_type")
    data_qubits = layout.data_qubits

    h_x = np.zeros((len(x_anc), layout.num_data_qubits), dtype=int)
    for k, anc in enumerate(x_anc):
        datas = layout.get_neighbors([anc])
        for data in datas:
            h_x[k, data_qubits.index(data)] = True

    h_z = np.zeros((len(z_anc), layout.num_data_qubits), dtype=int)
    for k, anc in enumerate(z_anc):
        datas = layout.get_neighbors([anc])
        for data in datas:
            h_z[k, data_qubits.index(data)] = True

    l_x = np.zeros((layout.num_logical_qubits, layout.num_data_qubits), dtype=int)
    for k, log in enumerate(layout.logical_qubits):
        datas = layout.logical_param("log_x", log)
        for data in datas:
            l_x[k, data_qubits.index(data)] = True

    l_z = np.zeros((layout.num_logical_qubits, layout.num_data_qubits), dtype=int)
    for k, log in enumerate(layout.logical_qubits):
        datas = layout.logical_param("log_z", log)
        for data in datas:
            l_z[k, data_qubits.index(data)] = True

    GF = galois.GF(2)
    h_x, h_z, l_x, l_z = GF(h_x), GF(h_z), GF(l_x), GF(l_z)

    # check 1. stabilisers commute with each other
    if not (h_x @ h_z.T == 0).all():
        raise ValueError("Stabilizers do not commute with each other.")

    # check 2. logical Paulis commute with all stabilisers
    if not (h_x @ l_z.T == 0).all():
        raise ValueError("Logical Pauli Zs anticommute with X stabilizers.")
    if not (h_z @ l_x.T == 0).all():
        raise ValueError("Logical Pauli Xs anticommute with Z stabilizers.")

    # check 3. logical Xi and Zi anticommute
    if not (l_x @ l_z.T == np.eye(layout.num_logical_qubits, dtype=int)).all():
        raise ValueError(
            "Logical Pauli X and Z do not follow the appropiate commutation relations"
        )

    # 4. logical Paulis are independent (i.e. no product of them is in the stabilizer group)
    hl_x = GF(np.concatenate([h_x, l_x], axis=0, dtype=int))
    if not (
        np.linalg.matrix_rank(hl_x)
        == np.linalg.matrix_rank(h_x) + layout.num_logical_qubits
    ):
        raise ValueError("A product of logical Pauli X is equivalent to stabilizer(s).")
    hl_z = GF(np.concatenate([h_z, l_z], axis=0, dtype=int))
    if not (
        np.linalg.matrix_rank(hl_z)
        == np.linalg.matrix_rank(h_z) + layout.num_logical_qubits
    ):
        raise ValueError("A product of logical Pauli Z is equivalent to stabilizer(s).")

    return

from collections.abc import Sequence
from copy import deepcopy

from ..layout import Layout
from ...log_gates.small_stellated_dodecahedron_code import (
    set_fold_trans_s,
    set_fold_trans_h,
    set_fold_trans_swap_r,
    set_fold_trans_swap_s,
    set_fold_trans_swap_a,
    set_fold_trans_swap_b,
    set_fold_trans_swap_c,
    set_idle,
)


def ssd_code(
    interaction_order: str | dict[str, Sequence[str]] = "parallel-6",
    define_trans_gates: bool = True,
) -> Layout:
    """Returns a layout for the Small Stellated Dodecahedron code.

    Parameters
    ----------
    interaction_order
        Name of the CNOT interaction order to perform in the QEC cycle.
        By default 'parallel-6'. The list of names can be found in
        ``INTERACTION_ORDERS``. It is possible to give directly the
        interaction order dictionary.
    define_trans_gates
        Flag for loading the parameters needed to run the transversal gates.
        By default ``True``.

    Returns
    -------
    layout
        Layout of the SSD code.

    Notes
    -----
    The qubit indexing and stabilizers follow:

        J. Q. Broshuis, "The Small Stellated Dodecahedron Code: Finding Interleaved
        Measurement Schedules", Bachelor's thesis.
        https://repository.tudelft.nl/record/uuid:4e6852c1-b18d-4b6b-8cc4-dc4587bff260

    Note that the stabilizers also follow:

        J. Conrad, C. Chamberland, N. P. Breuckmann, and B. M. Terhal,
        "The small stellated dodecahedron code and friends", Philosophical
        Transactions of the Royal Society A: Mathematical, Physical and Engineering
        Sciences 376, 20170323 (2018) arXiv:1712.07666 DOI

    The logicals correspond to the basis of subspaces V + W from:

        N. P. Breuckmann and S. Burton, "Fold-Transversal Clifford Gates for Quantum Codes",
        Quantum 8, 1372 (2024) arXiv:2202.06647 DOI

    where the basis of the subspace W has been modified to make it symplectic.
    Note that the stabilizer definition of Breuckmann is the same of Conrad
    except for a change in the stabilizer type (``X <-> Z``), thus the logicals
    have also been modified accordingly.
    The logical qubits ``"L1"`` to ``"L4"`` correspond to the logical subspace V
    and they have transversal gates that span the full 4-qubit Clifford group.
    """
    layout_dict = deepcopy(SSD_LAYOUT_DICT)

    if isinstance(interaction_order, str):
        if interaction_order not in INTERACTION_ORDERS:
            raise ValueError(
                f"'interaction_order' must be in {list(INTERACTION_ORDERS)}, "
                "but {interaction_order} was given."
            )
        layout_dict["interaction_order"] = INTERACTION_ORDERS[interaction_order]
    elif isinstance(interaction_order, dict):
        layout_dict["interaction_order"] = interaction_order
    else:
        raise TypeError(
            "'interaction_order' must be a str or a dict, "
            f"but {type(interaction_order)} was given."
        )

    layout = Layout(layout_dict)

    if define_trans_gates:
        set_fold_trans_s(layout)
        set_fold_trans_h(layout)
        set_fold_trans_swap_r(layout)
        set_fold_trans_swap_s(layout)
        set_fold_trans_swap_a(layout)
        set_fold_trans_swap_b(layout)
        set_fold_trans_swap_c(layout)
        set_idle(layout)

    return layout


INTERACTION_ORDERS = {
    "parallel-6": {
        "X1": ["D20", "D9", "D6", "D21", "D19", None],
        "X2": ["D10", "D22", "D4", "D1", "D13", None],
        "X3": ["D9", "D14", "D23", "D7", "D17", None],
        "X4": ["D13", "D11", "D20", "D18", "D24", None],
        "X5": ["D17", "D1", "D15", "D3", "D25", None],
        "X6": ["D4", "D18", "D27", "D30", "D2", None],
        "X7": ["D3", "D26", "D5", None, "D28", "D7"],
        "X8": ["D8", "D6", "D11", None, "D29", "D27"],
        "X9": ["D30", "D15", "D12", "D28", "D10", None],
        "X10": ["D26", "D19", "D29", "D16", "D14", None],
        "X11": ["D24", "D23", "D21", "D25", None, "D22"],
        "X12": ["D16", "D12", "D2", "D5", "D8", None],
        "Z1": ["D1", "D4", "D3", None, "D5", "D2"],
        "Z2": ["D5", None, "D9", "D8", "D7", "D6"],
        "Z3": ["D12", "D8", "D10", "D11", None, "D13"],
        "Z4": ["D14", "D17", "D16", "D12", "D15", None],
        "Z5": ["D18", "D20", None, "D2", "D16", "D19"],
        "Z6": ["D21", "D3", "D25", "D19", "D26", None],
        "Z7": ["D6", "D21", "D22", "D4", "D27", None],
        "Z8": ["D23", "D28", "D7", "D10", "D22", None],
        "Z9": ["D11", "D29", "D24", "D14", "D23", None],
        "Z10": [None, "D24", "D18", "D15", "D30", "D25"],
        "Z11": ["D27", "D30", "D26", "D29", None, "D28"],
        "Z12": [None, "D13", "D1", "D20", "D9", "D17"],
    }
}

SSD_LAYOUT_DICT = {
    "code": "small_stellated_dodecahedron_code",
    "logical_qubits": {
        "L1": {
            "ind": 0,
            "log_x": ["D23", "D4", "D5", "D6", "D28", "D29"],
            "log_z": ["D21", "D23", "D27", "D9", "D11", "D18"],
        },
        "L2": {
            "ind": 1,
            "log_x": ["D23", "D18", "D25", "D28", "D29", "D19"],
            "log_z": ["D21", "D23", "D26", "D5", "D9", "D16"],
        },
        "L3": {
            "ind": 2,
            "log_x": [
                "D12",
                "D19",
                "D23",
                "D4",
                "D6",
                "D5",
                "D17",
                "D25",
                "D13",
                "D28",
                "D29",
                "D18",
            ],
            "log_z": [
                "D7",
                "D17",
                "D24",
                "D21",
                "D27",
                "D22",
                "D2",
                "D3",
                "D13",
                "D30",
                "D12",
                "D9",
                "D16",
                "D11",
                "D6",
                "D29",
                "D19",
                "D23",
                "D26",
                "D5",
                "D18",
            ],
        },
        "L4": {
            "ind": 3,
            "log_x": [
                "D7",
                "D15",
                "D24",
                "D25",
                "D28",
                "D21",
                "D27",
                "D4",
                "D2",
                "D14",
                "D20",
                "D1",
                "D6",
                "D29",
                "D19",
                "D8",
                "D23",
                "D26",
                "D10",
                "D5",
                "D18",
            ],
            "log_z": [
                "D12",
                "D7",
                "D19",
                "D15",
                "D22",
                "D10",
                "D1",
                "D17",
                "D24",
                "D2",
                "D3",
                "D6",
                "D13",
                "D29",
                "D30",
            ],
        },
        "L5": {
            "ind": 4,
            "log_x": ["D20", "D15", "D1", "D24", "D2", "D14"],
            "log_z": ["D12", "D7", "D17", "D2", "D3", "D30"],
        },
        "L6": {
            "ind": 5,
            "log_x": ["D20", "D7", "D10", "D1", "D2", "D8"],
            "log_z": ["D12", "D22", "D24", "D2", "D13", "D30"],
        },
        "L7": {
            "ind": 6,
            "log_x": [
                "D21",
                "D20",
                "D7",
                "D27",
                "D15",
                "D26",
                "D10",
                "D1",
                "D24",
                "D2",
                "D14",
                "D8",
            ],
            "log_z": [
                "D7",
                "D15",
                "D17",
                "D24",
                "D21",
                "D27",
                "D22",
                "D2",
                "D3",
                "D13",
                "D30",
                "D12",
                "D1",
                "D9",
                "D16",
                "D11",
                "D23",
                "D26",
                "D10",
                "D5",
                "D18",
            ],
        },
        "L8": {
            "ind": 7,
            "log_x": [
                "D7",
                "D15",
                "D17",
                "D24",
                "D25",
                "D28",
                "D4",
                "D2",
                "D14",
                "D13",
                "D12",
                "D20",
                "D1",
                "D6",
                "D29",
                "D19",
                "D8",
                "D23",
                "D10",
                "D5",
                "D18",
            ],
            "log_z": [
                "D21",
                "D19",
                "D23",
                "D27",
                "D15",
                "D26",
                "D10",
                "D5",
                "D1",
                "D9",
                "D16",
                "D11",
                "D6",
                "D29",
                "D18",
            ],
        },
    },
    "distance": 3,
    "distance_x": 3,
    "distance_z": 3,
    "interaction_order": {},
    "layout": [
        {
            "qubit": "D1",
            "role": "data",
            "stab_type": None,
            "ind": 12,
            "neighbors": {"XA": "X2", "XB": "X5", "ZA": "Z1", "ZB": "Z12"},
            "coords": [-0.0025, -9.975],
        },
        {
            "qubit": "D2",
            "role": "data",
            "stab_type": None,
            "ind": 13,
            "neighbors": {"XA": "X6", "XB": "X12", "ZA": "Z1", "ZB": "Z5"},
            "coords": [-6.0175, -8.28],
        },
        {
            "qubit": "D3",
            "role": "data",
            "stab_type": None,
            "ind": 14,
            "neighbors": {"XA": "X5", "XB": "X7", "ZA": "Z1", "ZB": "Z6"},
            "coords": [-2.3925, -6.2725],
        },
        {
            "qubit": "D4",
            "role": "data",
            "stab_type": None,
            "ind": 15,
            "neighbors": {"XA": "X2", "XB": "X6", "ZA": "Z1", "ZB": "Z7"},
            "coords": [2.3925, -6.2725],
        },
        {
            "qubit": "D5",
            "role": "data",
            "stab_type": None,
            "ind": 16,
            "neighbors": {"XA": "X7", "XB": "X12", "ZA": "Z1", "ZB": "Z2"},
            "coords": [6.015, -8.28],
        },
        {
            "qubit": "D6",
            "role": "data",
            "stab_type": None,
            "ind": 17,
            "neighbors": {"XA": "X1", "XB": "X8", "ZA": "Z2", "ZB": "Z7"},
            "coords": [5.225, -4.215],
        },
        {
            "qubit": "D7",
            "role": "data",
            "stab_type": None,
            "ind": 18,
            "neighbors": {"XA": "X3", "XB": "X7", "ZA": "Z2", "ZB": "Z8"},
            "coords": [6.7025, 0.335],
        },
        {
            "qubit": "D8",
            "role": "data",
            "stab_type": None,
            "ind": 19,
            "neighbors": {"XA": "X8", "XB": "X12", "ZA": "Z2", "ZB": "Z3"},
            "coords": [9.7325, 3.1625],
        },
        {
            "qubit": "D9",
            "role": "data",
            "stab_type": None,
            "ind": 20,
            "neighbors": {"XA": "X1", "XB": "X3", "ZA": "Z2", "ZB": "Z12"},
            "coords": [9.485, -3.0825],
        },
        {
            "qubit": "D10",
            "role": "data",
            "stab_type": None,
            "ind": 21,
            "neighbors": {"XA": "X2", "XB": "X9", "ZA": "Z3", "ZB": "Z8"},
            "coords": [5.62, 3.67],
        },
        {
            "qubit": "D11",
            "role": "data",
            "stab_type": None,
            "ind": 22,
            "neighbors": {"XA": "X4", "XB": "X8", "ZA": "Z3", "ZB": "Z9"},
            "coords": [1.75, 6.4825],
        },
        {
            "qubit": "D12",
            "role": "data",
            "stab_type": None,
            "ind": 23,
            "neighbors": {"XA": "X9", "XB": "X12", "ZA": "Z3", "ZB": "Z4"},
            "coords": [0.0, 10.235],
        },
        {
            "qubit": "D13",
            "role": "data",
            "stab_type": None,
            "ind": 24,
            "neighbors": {"XA": "X2", "XB": "X4", "ZA": "Z3", "ZB": "Z12"},
            "coords": [5.8625, 8.07],
        },
        {
            "qubit": "D14",
            "role": "data",
            "stab_type": None,
            "ind": 25,
            "neighbors": {"XA": "X3", "XB": "X10", "ZA": "Z4", "ZB": "Z9"},
            "coords": [-1.75, 6.4825],
        },
        {
            "qubit": "D15",
            "role": "data",
            "stab_type": None,
            "ind": 26,
            "neighbors": {"XA": "X5", "XB": "X9", "ZA": "Z4", "ZB": "Z10"},
            "coords": [-5.62, 3.67],
        },
        {
            "qubit": "D16",
            "role": "data",
            "stab_type": None,
            "ind": 27,
            "neighbors": {"XA": "X10", "XB": "X12", "ZA": "Z4", "ZB": "Z5"},
            "coords": [-9.7325, 3.1625],
        },
        {
            "qubit": "D17",
            "role": "data",
            "stab_type": None,
            "ind": 28,
            "neighbors": {"XA": "X3", "XB": "X5", "ZA": "Z4", "ZB": "Z12"},
            "coords": [-5.8625, 8.07],
        },
        {
            "qubit": "D18",
            "role": "data",
            "stab_type": None,
            "ind": 29,
            "neighbors": {"XA": "X4", "XB": "X6", "ZA": "Z5", "ZB": "Z10"},
            "coords": [-6.705, 0.3375],
        },
        {
            "qubit": "D19",
            "role": "data",
            "stab_type": None,
            "ind": 30,
            "neighbors": {"XA": "X1", "XB": "X10", "ZA": "Z5", "ZB": "Z6"},
            "coords": [-5.2275, -4.2125],
        },
        {
            "qubit": "D20",
            "role": "data",
            "stab_type": None,
            "ind": 31,
            "neighbors": {"XA": "X1", "XB": "X4", "ZA": "Z5", "ZB": "Z12"},
            "coords": [-9.4875, -3.0825],
        },
        {
            "qubit": "D21",
            "role": "data",
            "stab_type": None,
            "ind": 32,
            "neighbors": {"XA": "X1", "XB": "X11", "ZA": "Z6", "ZB": "Z7"},
            "coords": [0.0, -4.265],
        },
        {
            "qubit": "D22",
            "role": "data",
            "stab_type": None,
            "ind": 33,
            "neighbors": {"XA": "X2", "XB": "X11", "ZA": "Z7", "ZB": "Z8"},
            "coords": [4.055, -1.32],
        },
        {
            "qubit": "D23",
            "role": "data",
            "stab_type": None,
            "ind": 34,
            "neighbors": {"XA": "X3", "XB": "X11", "ZA": "Z8", "ZB": "Z9"},
            "coords": [2.505, 3.4525],
        },
        {
            "qubit": "D24",
            "role": "data",
            "stab_type": None,
            "ind": 35,
            "neighbors": {"XA": "X4", "XB": "X11", "ZA": "Z9", "ZB": "Z10"},
            "coords": [-2.505, 3.4525],
        },
        {
            "qubit": "D25",
            "role": "data",
            "stab_type": None,
            "ind": 36,
            "neighbors": {"XA": "X5", "XB": "X11", "ZA": "Z6", "ZB": "Z10"},
            "coords": [-4.0575, -1.3175],
        },
        {
            "qubit": "D26",
            "role": "data",
            "stab_type": None,
            "ind": 37,
            "neighbors": {"XA": "X7", "XB": "X10", "ZA": "Z6", "ZB": "Z11"},
            "coords": [-1.1225, -1.6825],
        },
        {
            "qubit": "D27",
            "role": "data",
            "stab_type": None,
            "ind": 38,
            "neighbors": {"XA": "X6", "XB": "X8", "ZA": "Z7", "ZB": "Z11"},
            "coords": [1.2225, -1.6825],
        },
        {
            "qubit": "D28",
            "role": "data",
            "stab_type": None,
            "ind": 39,
            "neighbors": {"XA": "X7", "XB": "X9", "ZA": "Z8", "ZB": "Z11"},
            "coords": [1.9775, 0.6425],
        },
        {
            "qubit": "D29",
            "role": "data",
            "stab_type": None,
            "ind": 40,
            "neighbors": {"XA": "X8", "XB": "X10", "ZA": "Z9", "ZB": "Z11"},
            "coords": [0.0, 2.08],
        },
        {
            "qubit": "D30",
            "role": "data",
            "stab_type": None,
            "ind": 41,
            "neighbors": {"XA": "X6", "XB": "X9", "ZA": "Z10", "ZB": "Z11"},
            "coords": [-1.9775, 0.6425],
        },
        {
            "qubit": "X1",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 42,
            "neighbors": {
                "DA": "D6",
                "DB": "D9",
                "DC": "D19",
                "DD": "D20",
                "DE": "D21",
            },
            "coords": [0.25, -7.566],
        },
        {
            "qubit": "X2",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 43,
            "neighbors": {
                "DA": "D1",
                "DB": "D4",
                "DC": "D10",
                "DD": "D13",
                "DE": "D22",
            },
            "coords": [7.682, -2.166],
        },
        {
            "qubit": "X3",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 44,
            "neighbors": {
                "DA": "D7",
                "DB": "D9",
                "DC": "D14",
                "DD": "D17",
                "DE": "D23",
            },
            "coords": [4.843, 6.574],
        },
        {
            "qubit": "X4",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 45,
            "neighbors": {
                "DA": "D11",
                "DB": "D13",
                "DC": "D18",
                "DD": "D20",
                "DE": "D24",
            },
            "coords": [-4.343, 6.574],
        },
        {
            "qubit": "X5",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 46,
            "neighbors": {
                "DA": "D1",
                "DB": "D3",
                "DC": "D15",
                "DD": "D17",
                "DE": "D25",
            },
            "coords": [-7.182, -2.166],
        },
        {
            "qubit": "X6",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 47,
            "neighbors": {
                "DA": "D2",
                "DB": "D4",
                "DC": "D18",
                "DD": "D27",
                "DE": "D30",
            },
            "coords": [-2.33, -3.3],
        },
        {
            "qubit": "X7",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 48,
            "neighbors": {"DA": "D3", "DB": "D5", "DC": "D7", "DD": "D26", "DE": "D28"},
            "coords": [2.83, -3.3],
        },
        {
            "qubit": "X8",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 49,
            "neighbors": {
                "DA": "D6",
                "DB": "D8",
                "DC": "D11",
                "DD": "D27",
                "DE": "D29",
            },
            "coords": [4.423, 1.606],
        },
        {
            "qubit": "X9",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 50,
            "neighbors": {
                "DA": "D10",
                "DB": "D12",
                "DC": "D15",
                "DD": "D28",
                "DE": "D30",
            },
            "coords": [0.25, 4.64],
        },
        {
            "qubit": "X10",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 51,
            "neighbors": {
                "DA": "D14",
                "DB": "D16",
                "DC": "D19",
                "DD": "D26",
                "DE": "D29",
            },
            "coords": [-3.923, 1.606],
        },
        {
            "qubit": "X11",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 52,
            "neighbors": {
                "DA": "D21",
                "DB": "D22",
                "DC": "D23",
                "DD": "D24",
                "DE": "D25",
            },
            "coords": [0.5, 0.0],
        },
        {
            "qubit": "X12",
            "role": "anc",
            "stab_type": "x_type",
            "ind": 53,
            "neighbors": {"DA": "D2", "DB": "D5", "DC": "D8", "DD": "D12", "DE": "D16"},
            "coords": [0.0, 0.5],
        },
        {
            "qubit": "Z1",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 0,
            "neighbors": {"DA": "D1", "DB": "D2", "DC": "D3", "DD": "D4", "DE": "D5"},
            "coords": [-0.25, -8.066],
        },
        {
            "qubit": "Z2",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 1,
            "neighbors": {"DA": "D5", "DB": "D6", "DC": "D7", "DD": "D8", "DE": "D9"},
            "coords": [7.182, -2.666],
        },
        {
            "qubit": "Z3",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 2,
            "neighbors": {
                "DA": "D8",
                "DB": "D10",
                "DC": "D11",
                "DD": "D12",
                "DE": "D13",
            },
            "coords": [4.343, 6.074],
        },
        {
            "qubit": "Z4",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 3,
            "neighbors": {
                "DA": "D12",
                "DB": "D14",
                "DC": "D15",
                "DD": "D16",
                "DE": "D17",
            },
            "coords": [-4.843, 6.074],
        },
        {
            "qubit": "Z5",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 4,
            "neighbors": {
                "DA": "D2",
                "DB": "D16",
                "DC": "D18",
                "DD": "D19",
                "DE": "D20",
            },
            "coords": [-7.682, -2.666],
        },
        {
            "qubit": "Z6",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 5,
            "neighbors": {
                "DA": "D3",
                "DB": "D19",
                "DC": "D21",
                "DD": "D25",
                "DE": "D26",
            },
            "coords": [-2.83, -3.8],
        },
        {
            "qubit": "Z7",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 6,
            "neighbors": {
                "DA": "D4",
                "DB": "D6",
                "DC": "D21",
                "DD": "D22",
                "DE": "D27",
            },
            "coords": [2.33, -3.8],
        },
        {
            "qubit": "Z8",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 7,
            "neighbors": {
                "DA": "D7",
                "DB": "D10",
                "DC": "D22",
                "DD": "D23",
                "DE": "D28",
            },
            "coords": [3.923, 1.106],
        },
        {
            "qubit": "Z9",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 8,
            "neighbors": {
                "DA": "D11",
                "DB": "D14",
                "DC": "D23",
                "DD": "D24",
                "DE": "D29",
            },
            "coords": [-0.25, 4.14],
        },
        {
            "qubit": "Z10",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 9,
            "neighbors": {
                "DA": "D15",
                "DB": "D18",
                "DC": "D24",
                "DD": "D25",
                "DE": "D30",
            },
            "coords": [-4.423, 1.106],
        },
        {
            "qubit": "Z11",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 10,
            "neighbors": {
                "DA": "D26",
                "DB": "D27",
                "DC": "D28",
                "DD": "D29",
                "DE": "D30",
            },
            "coords": [0.0, -0.5],
        },
        {
            "qubit": "Z12",
            "role": "anc",
            "stab_type": "z_type",
            "ind": 11,
            "neighbors": {
                "DA": "D1",
                "DB": "D9",
                "DC": "D13",
                "DD": "D17",
                "DE": "D20",
            },
            "coords": [-0.5, 0.0],
        },
    ],
}

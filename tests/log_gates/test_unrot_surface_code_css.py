from surface_sim.detectors import get_new_stab_dict_from_layout
from surface_sim.layouts import unrot_surface_code
from surface_sim.log_gates.unrot_surface_code_css import (
    set_encoding,
    set_fold_trans_h,
    set_fold_trans_s,
)


def test_set_fold_trans_s():
    layout = unrot_surface_code(distance=3)
    set_fold_trans_s(layout, "D1")
    gate_label = f"log_fold_trans_s_{layout.logical_qubits[0]}"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout.param(gate_label, x_stab)[key] for x_stab in x_stabs]
        assert new_stab_x == [
            ["X1", "Z1"],
            ["X2", "Z4"],
            ["X3", "Z2"],
            ["X4", "Z5"],
            ["X5", "Z3"],
            ["X6", "Z6"],
        ]

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout.param(gate_label, z_stab)[key] for z_stab in z_stabs]
        assert new_stab_z == [["Z1"], ["Z2"], ["Z3"], ["Z4"], ["Z5"], ["Z6"]]

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    cz_gates = [layout.param(gate_label, d)["cz"] for d in data_qubits]
    assert cz_gates == [
        None,
        "D6",
        "D11",
        None,
        "D9",
        "D2",
        None,
        "D12",
        "D5",
        None,
        "D3",
        "D8",
        None,
    ]

    local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
    assert local_gates == [
        "S",
        "I",
        "I",
        "S_DAG",
        "I",
        "I",
        "S",
        "I",
        "I",
        "S_DAG",
        "I",
        "I",
        "S",
    ]

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == new_stabs_inv
    for z in z_stabs:
        assert len(new_stabs[z]) == 1
    for x in x_stabs:
        assert len(new_stabs[x]) == 2

    return


def test_set_fold_trans_h():
    layout = unrot_surface_code(distance=3)
    set_fold_trans_h(layout, "D1")
    gate_label = f"log_fold_trans_h_{layout.logical_qubits[0]}"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout.param(gate_label, x_stab)[key] for x_stab in x_stabs]
        assert new_stab_x == [["Z1"], ["Z4"], ["Z2"], ["Z5"], ["Z3"], ["Z6"]]

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout.param(gate_label, z_stab)[key] for z_stab in z_stabs]
        assert new_stab_z == [["X1"], ["X3"], ["X5"], ["X2"], ["X4"], ["X6"]]

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap"] for d in data_qubits]
    assert swap_gates == [
        None,
        "D6",
        "D11",
        None,
        "D9",
        "D2",
        None,
        "D12",
        "D5",
        None,
        "D3",
        "D8",
        None,
    ]

    local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
    assert set(local_gates) == set(["H"])

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == new_stabs_inv
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return


def test_set_encoding():
    layout = unrot_surface_code(4)
    set_encoding(layout)
    gate_label = f"encoding_{layout.logical_qubits[0]}"

    data_qubits = sorted(layout.data_qubits)
    labels = sorted([layout.param(gate_label, d)["label"] for d in data_qubits])
    print(labels)
    assert labels == [
        (0, 0),
        (0, 2),
        (0, 4),
        (0, 6),
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 0),
        (2, 2),
        (2, 4),
        (2, 6),
        (3, 1),
        (3, 3),
        (3, 5),
        (4, 0),
        (4, 2),
        (4, 4),
        (4, 6),
        (5, 1),
        (5, 3),
        (5, 5),
        (6, 0),
        (6, 2),
        (6, 4),
        (6, 6),
    ]

    layout = unrot_surface_code(3)
    set_encoding(layout)
    gate_label = f"encoding_{layout.logical_qubits[0]}"

    data_qubits = sorted(layout.data_qubits)
    labels = sorted([layout.param(gate_label, d)["label"] for d in data_qubits])
    assert labels == [
        (0, 0),
        (0, 2),
        (0, 4),
        (1, 1),
        (1, 3),
        (2, 0),
        (2, 2),
        (2, 4),
        (3, 1),
        (3, 3),
        (4, 0),
        (4, 2),
        (4, 4),
    ]

    return

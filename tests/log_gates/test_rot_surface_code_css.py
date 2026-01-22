from surface_sim.log_gates.rot_surface_code_css import (
    set_fold_trans_s,
    set_trans_cnot_mid_cycle_css,
    set_encoding,
)
from surface_sim.layouts import rot_surface_code_rectangle, rot_surface_code
from surface_sim.detectors import get_new_stab_dict_from_layout


def test_set_trans_cnot_mid_cycle_css():
    distance = 3
    layout_c = rot_surface_code(distance=distance)
    layout_t = rot_surface_code(
        distance=distance,
        logical_qubit_label="L1",
        init_point=(1000, 1000),
        init_data_qubit_id=1000,
        init_zanc_qubit_id=1000,
        init_xanc_qubit_id=1000,
        init_ind=layout_c.get_max_ind() + 1,
    )
    set_trans_cnot_mid_cycle_css(layout_c, layout_t)
    gate_label = f"log_trans_cnot_mid_cycle_css_{layout_c.logical_qubits[0]}_{layout_t.logical_qubits[0]}"

    x_stabs = layout_c.get_qubits(role="anc", stab_type="x_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout_c.param(gate_label, x_stab)[key] for x_stab in x_stabs]
        assert new_stab_x == [[i, f"X{int(i[1:]) + 1000 - 1}"] for i in x_stabs]

    x_stabs = layout_t.get_qubits(role="anc", stab_type="x_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout_t.param(gate_label, x_stab)[key] for x_stab in x_stabs]
        assert new_stab_x == [[i] for i in x_stabs]

    z_stabs = layout_t.get_qubits(role="anc", stab_type="z_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout_t.param(gate_label, z_stab)[key] for z_stab in z_stabs]
        assert new_stab_z == [[i, f"Z{int(i[1:]) - 1000 + 1}"] for i in z_stabs]

    z_stabs = layout_c.get_qubits(role="anc", stab_type="z_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout_c.param(gate_label, z_stab)[key] for z_stab in z_stabs]
        assert new_stab_z == [[i] for i in z_stabs]

    data_qubits = layout_c.data_qubits
    x_stabs = layout_c.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout_c.get_qubits(role="anc", stab_type="z_type")
    for n, qs in zip(["D", "X", "Z"], [data_qubits, x_stabs, z_stabs]):
        cnot_gates = [layout_c.param(gate_label, q)["cnot"] for q in qs]
        assert cnot_gates == [f"{n}{int(i[1:]) + 1000 - 1}" for i in qs]

    data_qubits = layout_t.data_qubits
    x_stabs = layout_t.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout_t.get_qubits(role="anc", stab_type="z_type")
    for n, qs in zip(["D", "X", "Z"], [data_qubits, x_stabs, z_stabs]):
        cnot_gates = [layout_t.param(gate_label, q)["cnot"] for q in qs]
        assert cnot_gates == [f"{n}{int(i[1:]) - 1000 + 1}" for i in qs]

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout_c, gate_label)
    assert new_stabs == new_stabs_inv
    x_stabs = layout_c.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout_c.get_qubits(role="anc", stab_type="z_type")
    for z in z_stabs:
        assert len(new_stabs[z]) == 1
    for x in x_stabs:
        assert len(new_stabs[x]) == 2
    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout_t, gate_label)
    assert new_stabs == new_stabs_inv
    x_stabs = layout_t.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout_t.get_qubits(role="anc", stab_type="z_type")
    for z in z_stabs:
        assert len(new_stabs[z]) == 2
    for x in x_stabs:
        assert len(new_stabs[x]) == 1

    return


def test_set_fold_trans_s():
    layout = rot_surface_code_rectangle(distance_z=4, distance_x=3)
    set_fold_trans_s(layout, "D1")
    gate_label = f"log_fold_trans_s_{layout.logical_qubits[0]}"

    x_stab = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout.param(gate_label, x_stab)[key] for x_stab in x_stab]
        assert new_stab_x == [
            ["X1"],
            ["X2", "Z3"],
            ["X3", "Z1"],
            ["X4", "Z2"],
            ["X5", "Z4"],
            ["X6", "Z5"],
        ]

    z_stab = sorted(layout.get_qubits(role="anc", stab_type="z_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout.param(gate_label, z_stab)[key] for z_stab in z_stab]
        assert new_stab_z == [["Z1"], ["Z2"], ["Z3"], ["Z4"], ["Z5"]]

    data_qubits = sorted(layout.data_qubits)
    cz_gates = [layout.param(gate_label, d)["cz"] for d in data_qubits]
    assert cz_gates == [
        "D2",
        "D8",
        "D12",
        "D11",
        "D1",
        "D5",
        "D9",
        "D3",
        "D7",
        "D6",
        "D10",
        "D4",
    ]

    local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
    assert local_gates == [
        "S",
        "I",
        "I",
        "S_DAG",
        "S",
        "I",
        "I",
        "I",
        "I",
        "S",
        "I",
        "I",
    ]

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == new_stabs_inv
    for z in z_stab:
        assert len(new_stabs[z]) == 1
    for x in x_stab:
        if x == "X1":
            assert len(new_stabs[x]) == 1
        else:
            assert len(new_stabs[x]) == 2
    return


def test_set_encoding():
    layout = rot_surface_code(4)
    set_encoding(layout)
    gate_label = f"encoding_{layout.logical_qubits[0]}"

    data_qubits = sorted(layout.data_qubits)
    labels = [layout.param(gate_label, d)["label"] for d in data_qubits]
    assert labels == [
        (1, 0),
        (0, 1),
        (0, 2),
        (1, 7),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 11),
        (1, 10),
        (1, 9),
        (1, 1),
        (0, 0),
        (0, 3),
        (1, 8),
        (1, 2),
    ]

    return

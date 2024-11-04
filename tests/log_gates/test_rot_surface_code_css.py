from surface_sim.log_gates.rot_surface_code_css import set_trans_s, set_trans_cnot
from surface_sim.layouts import rot_surface_code_rectangle, rot_surface_code
from surface_sim.detectors import get_new_stab_dict_from_layout


def test_set_trans_s():
    layout = rot_surface_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    gate_label = f"trans-s_{layout.get_logical_qubits()[0]}"

    x_stab = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    new_stab_x = [layout.param(gate_label, x_stab)["new_stab_gen"] for x_stab in x_stab]
    assert new_stab_x == [
        ["X1"],
        ["X2", "Z3"],
        ["X3", "Z1"],
        ["X4", "Z2"],
        ["X5", "Z4"],
        ["X6", "Z5"],
    ]

    z_stab = sorted(layout.get_qubits(role="anc", stab_type="z_type"))
    new_stab_z = [layout.param(gate_label, z_stab)["new_stab_gen"] for z_stab in z_stab]
    assert new_stab_z == [["Z1"], ["Z2"], ["Z3"], ["Z4"], ["Z5"]]

    data_qubits = sorted(layout.get_qubits(role="data"))
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

    new_stabs = get_new_stab_dict_from_layout(layout, gate_label)
    for z in z_stab:
        assert len(new_stabs[z]) == 1
    for x in x_stab:
        if x == "X1":
            assert len(new_stabs[x]) == 1
        else:
            assert len(new_stabs[x]) == 2
    return


def test_set_trans_cnot():
    layout_c = rot_surface_code(distance=3)
    layout_t = rot_surface_code(
        distance=3,
        logical_qubit_label="L1",
        init_point=(20, 20),
        init_data_qubit_id=20,
        init_zanc_qubit_id=9,
        init_xanc_qubit_id=9,
        init_ind=layout_c.get_max_ind() + 1,
    )
    set_trans_cnot(layout_c, layout_t)
    gate_label = f"trans-cnot_{layout_c.get_logical_qubits()[0]}_{layout_t.get_logical_qubits()[0]}"

    x_stabs = layout_c.get_qubits(role="anc", stab_type="x_type")
    new_stab_x = [
        layout_c.param(gate_label, x_stab)["new_stab_gen"] for x_stab in x_stabs
    ]
    assert new_stab_x == [[i, f"X{int(i[1:])+9-1}"] for i in x_stabs]

    x_stabs = layout_t.get_qubits(role="anc", stab_type="x_type")
    new_stab_x = [
        layout_t.param(gate_label, x_stab)["new_stab_gen"] for x_stab in x_stabs
    ]
    assert new_stab_x == [[i] for i in x_stabs]

    z_stabs = layout_t.get_qubits(role="anc", stab_type="z_type")
    new_stab_z = [
        layout_t.param(gate_label, z_stab)["new_stab_gen"] for z_stab in z_stabs
    ]
    assert new_stab_z == [[i, f"Z{int(i[1:])-9+1}"] for i in z_stabs]

    z_stabs = layout_c.get_qubits(role="anc", stab_type="z_type")
    new_stab_z = [
        layout_c.param(gate_label, z_stab)["new_stab_gen"] for z_stab in z_stabs
    ]
    assert new_stab_z == [[i] for i in z_stabs]

    data_qubits = layout_c.get_qubits(role="data")
    cz_gates = [layout_c.param(gate_label, d)["cnot"] for d in data_qubits]
    assert cz_gates == [f"D{int(i[1:])+20-1}" for i in data_qubits]

    new_stabs = get_new_stab_dict_from_layout(layout_c, gate_label)
    x_stabs = layout_c.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout_c.get_qubits(role="anc", stab_type="z_type")
    for z in z_stabs:
        assert len(new_stabs[z]) == 1
    for x in x_stabs:
        assert len(new_stabs[x]) == 2
    new_stabs = get_new_stab_dict_from_layout(layout_t, gate_label)
    x_stabs = layout_t.get_qubits(role="anc", stab_type="x_type")
    z_stabs = layout_t.get_qubits(role="anc", stab_type="z_type")
    for z in z_stabs:
        assert len(new_stabs[z]) == 2
    for x in x_stabs:
        assert len(new_stabs[x]) == 1

    return

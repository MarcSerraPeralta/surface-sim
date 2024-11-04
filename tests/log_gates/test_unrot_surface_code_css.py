from surface_sim.log_gates.unrot_surface_code_css import (
    set_trans_s,
    set_trans_h,
    set_trans_cnot,
)
from surface_sim.layouts import unrot_surface_code
from surface_sim.detectors import get_new_stab_dict_from_layout


def test_set_trans_s():
    layout = unrot_surface_code(distance=3)
    set_trans_s(layout, "D1")
    gate_label = f"trans-s_{layout.get_logical_qubits()[0]}"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    new_stab_x = [
        layout.param(gate_label, x_stab)["new_stab_gen"] for x_stab in x_stabs
    ]
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
    new_stab_z = [
        layout.param(gate_label, z_stab)["new_stab_gen"] for z_stab in z_stabs
    ]
    assert new_stab_z == [["Z1"], ["Z2"], ["Z3"], ["Z4"], ["Z5"], ["Z6"]]

    data_qubits = sorted(layout.get_qubits(role="data"), key=lambda x: int(x[1:]))
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

    new_stabs = get_new_stab_dict_from_layout(layout, gate_label)
    for z in z_stabs:
        assert len(new_stabs[z]) == 1
    for x in x_stabs:
        assert len(new_stabs[x]) == 2

    return


def test_set_trans_cnot():
    layout_c = unrot_surface_code(distance=3)
    layout_t = unrot_surface_code(
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


def test_set_trans_h():
    layout = unrot_surface_code(distance=3)
    set_trans_h(layout, "D1")
    gate_label = f"trans-h_{layout.get_logical_qubits()[0]}"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    new_stab_x = [
        layout.param(gate_label, x_stab)["new_stab_gen"] for x_stab in x_stabs
    ]
    assert new_stab_x == [["Z1"], ["Z4"], ["Z2"], ["Z5"], ["Z3"], ["Z6"]]

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    new_stab_z = [
        layout.param(gate_label, z_stab)["new_stab_gen"] for z_stab in z_stabs
    ]
    assert new_stab_z == [["X1"], ["X3"], ["X5"], ["X2"], ["X4"], ["X6"]]

    data_qubits = sorted(layout.get_qubits(role="data"), key=lambda x: int(x[1:]))
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

    new_stabs = get_new_stab_dict_from_layout(layout, gate_label)
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return

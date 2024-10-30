from surface_sim.log_gates.rot_surface_code_css import set_trans_s
from surface_sim.layouts import rot_surface_code_rectangle
from surface_sim.detectors import get_new_stab_dict_from_layout


def test_set_trans_s():
    layout = rot_surface_code_rectangle(distance_z=4, distance_x=3)
    set_trans_s(layout, "D1")
    gate_label = f"trans_s_{layout.get_logical_qubits()[0]}"

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

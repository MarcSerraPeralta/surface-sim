from surface_sim.log_gates.unrot_surface_code_css import set_trans_s
from surface_sim.layouts import unrot_surf_code


def test_set_trans_s():
    layout = unrot_surf_code(distance=3)
    print(layout.get_qubits(role="data"))
    set_trans_s(layout, "D1")
    gate_label = f"trans_s_{layout.get_logical_qubits()[0]}"

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

    stab_gen_matrix = layout.stab_gen_matrix(gate_label)
    for z in z_stabs:
        assert stab_gen_matrix.sel(new_stab_gen=z).sum() == 1
    for x in x_stabs:
        assert stab_gen_matrix.sel(new_stab_gen=x).sum() == 2
    return

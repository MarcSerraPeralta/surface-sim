from surface_sim.detectors import get_new_stab_dict_from_layout
from surface_sim.layouts import ssd_code
from surface_sim.log_gates.small_stellated_dodecahedron_code import (
    set_fold_trans_h,
    set_fold_trans_s,
    set_fold_trans_swap_a,
    set_fold_trans_swap_b,
    set_fold_trans_swap_c,
    set_fold_trans_swap_r,
    set_fold_trans_swap_s,
)


def test_set_fold_trans_s():
    layout = ssd_code()
    set_fold_trans_s(layout)
    gate_label = "log_fold_trans_s"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 2 and set(types) == set(["X", "Z"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["Z"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    cz_gates = [layout.param(gate_label, d)["cz"] for d in data_qubits]
    assert len([q for q in cz_gates if q is not None]) == 30 - 6

    local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
    assert len([q for q in local_gates if q == "S"]) == 3
    assert len([q for q in local_gates if q == "S_DAG"]) == 3
    assert len([q for q in local_gates if q is None]) == 30 - 6

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == new_stabs_inv
    for z in z_stabs:
        assert len(new_stabs[z]) == 1
    for x in x_stabs:
        assert len(new_stabs[x]) == 2

    return


def test_set_fold_trans_h():
    layout = ssd_code()
    set_fold_trans_h(layout)
    gate_label = "log_fold_trans_h"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 1 and set(types) == set(["Z"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["X"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 6

    local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
    assert set(local_gates) == set(["H"])

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == new_stabs_inv
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return


def test_set_fold_trans_swap_r():
    layout = ssd_code()
    set_fold_trans_swap_r(layout)
    gate_label = "log_fold_trans_swap_r"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 1 and set(types) == set(["X"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["Z"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap_1"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4
    swap_gates = [layout.param(gate_label, d)["swap_2"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == {list(v)[0]: {k} for k, v in new_stabs_inv.items()}
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return


def test_set_fold_trans_swap_s():
    layout = ssd_code()
    set_fold_trans_swap_s(layout)
    gate_label = "log_fold_trans_swap_s"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 1 and set(types) == set(["X"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["Z"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap_1"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4
    swap_gates = [layout.param(gate_label, d)["swap_2"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == {list(v)[0]: {k} for k, v in new_stabs_inv.items()}
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return


def test_set_fold_trans_swap_a():
    layout = ssd_code()
    set_fold_trans_swap_a(layout)
    gate_label = "log_fold_trans_swap_a"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 1 and set(types) == set(["X"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["Z"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == {list(v)[0]: {k} for k, v in new_stabs_inv.items()}
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return


def test_set_fold_trans_swap_b():
    layout = ssd_code()
    set_fold_trans_swap_b(layout)
    gate_label = "log_fold_trans_swap_b"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 1 and set(types) == set(["X"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["Z"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == {list(v)[0]: {k} for k, v in new_stabs_inv.items()}
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return


def test_set_fold_trans_swap_c():
    layout = ssd_code()
    set_fold_trans_swap_c(layout)
    gate_label = "log_fold_trans_swap_c"

    x_stabs = sorted(layout.get_qubits(role="anc", stab_type="x_type"))
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for x_stab in x_stabs:
            new_stabs_x = layout.param(gate_label, x_stab)[key]
            types = [s[0] for s in new_stabs_x]
            assert len(types) == 1 and set(types) == set(["X"])

    z_stabs = sorted(
        layout.get_qubits(role="anc", stab_type="z_type"), key=lambda x: int(x[1:])
    )
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        for z_stab in z_stabs:
            new_stabs_z = layout.param(gate_label, z_stab)[key]
            types = [s[0] for s in new_stabs_z]
            assert len(types) == 1 and set(types) == set(["Z"])

    data_qubits = sorted(layout.data_qubits, key=lambda x: int(x[1:]))
    swap_gates = [layout.param(gate_label, d)["swap"] for d in data_qubits]
    assert len([q for q in swap_gates if q is not None]) == 30 - 4

    new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
    assert new_stabs == {list(v)[0]: {k} for k, v in new_stabs_inv.items()}
    for s in z_stabs + x_stabs:
        assert len(new_stabs[s]) == 1

    return

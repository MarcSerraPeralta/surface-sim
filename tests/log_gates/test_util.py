from surface_sim.log_gates.util import set_x, set_z, set_idle, set_trans_cnot
from surface_sim.layouts import unrot_surface_code, rot_surface_code
from surface_sim.detectors import get_new_stab_dict_from_layout


def test_set_x():
    for layout_constructor in [unrot_surface_code, rot_surface_code]:
        layout = layout_constructor(distance=3)
        set_x(layout)
        gate_label = f"log_x_{layout.get_logical_qubits()[0]}"

        stabs = layout.get_qubits(role="anc")
        for key in ["new_stab_gen", "new_stab_gen_inv"]:
            new_stabs = [layout.param(gate_label, s)[key] for s in stabs]
            assert new_stabs == [[s] for s in stabs]

        data_qubits = layout.get_qubits(role="data")
        local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
        assert len([g for g in local_gates if g == "X"]) % 2 == 1

        new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
        assert new_stabs == new_stabs_inv
        for s in stabs:
            assert new_stabs[s] == {s}

    return


def test_set_z():
    for layout_constructor in [unrot_surface_code, rot_surface_code]:
        layout = layout_constructor(distance=3)
        set_z(layout)
        gate_label = f"log_z_{layout.get_logical_qubits()[0]}"

        stabs = layout.get_qubits(role="anc")
        for key in ["new_stab_gen", "new_stab_gen_inv"]:
            new_stabs = [layout.param(gate_label, s)[key] for s in stabs]
            assert new_stabs == [[s] for s in stabs]

        data_qubits = layout.get_qubits(role="data")
        local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
        assert len([g for g in local_gates if g == "Z"]) % 2 == 1

        new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
        assert new_stabs == new_stabs_inv
        for s in stabs:
            assert new_stabs[s] == {s}

    return


def test_set_idle():
    for layout_constructor in [unrot_surface_code, rot_surface_code]:
        layout = layout_constructor(distance=3)
        set_idle(layout)
        gate_label = f"idle_{layout.get_logical_qubits()[0]}"

        stabs = layout.get_qubits(role="anc")
        for key in ["new_stab_gen", "new_stab_gen_inv"]:
            new_stabs = [layout.param(gate_label, s)[key] for s in stabs]
            assert new_stabs == [[s] for s in stabs]

        data_qubits = layout.get_qubits(role="data")
        local_gates = [layout.param(gate_label, d)["local"] for d in data_qubits]
        assert len([g for g in local_gates if g != "I"]) == 0

        new_stabs, new_stabs_inv = get_new_stab_dict_from_layout(layout, gate_label)
        assert new_stabs == new_stabs_inv
        for s in stabs:
            assert new_stabs[s] == {s}

    return


def test_set_trans_cnot():
    for layout_constructor in [unrot_surface_code, rot_surface_code]:
        distance = 3
        layout_c = layout_constructor(distance=distance)
        layout_t = layout_constructor(
            distance=distance,
            logical_qubit_label="L1",
            init_point=(1000, 1000),
            init_data_qubit_id=1000,
            init_zanc_qubit_id=1000,
            init_xanc_qubit_id=1000,
            init_ind=layout_c.get_max_ind() + 1,
        )
        set_trans_cnot(layout_c, layout_t)
        gate_label = f"log_trans_cnot_{layout_c.get_logical_qubits()[0]}_{layout_t.get_logical_qubits()[0]}"

    x_stabs = layout_c.get_qubits(role="anc", stab_type="x_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout_c.param(gate_label, x_stab)[key] for x_stab in x_stabs]
        assert new_stab_x == [[i, f"X{int(i[1:])+1000-1}"] for i in x_stabs]

    x_stabs = layout_t.get_qubits(role="anc", stab_type="x_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_x = [layout_t.param(gate_label, x_stab)[key] for x_stab in x_stabs]
        assert new_stab_x == [[i] for i in x_stabs]

    z_stabs = layout_t.get_qubits(role="anc", stab_type="z_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout_t.param(gate_label, z_stab)[key] for z_stab in z_stabs]
        assert new_stab_z == [[i, f"Z{int(i[1:])-1000+1}"] for i in z_stabs]

    z_stabs = layout_c.get_qubits(role="anc", stab_type="z_type")
    for key in ["new_stab_gen", "new_stab_gen_inv"]:
        new_stab_z = [layout_c.param(gate_label, z_stab)[key] for z_stab in z_stabs]
        assert new_stab_z == [[i] for i in z_stabs]

    data_qubits = layout_c.get_qubits(role="data")
    cz_gates = [layout_c.param(gate_label, d)["cz"] for d in data_qubits]
    assert cz_gates == [f"D{int(i[1:])+1000-1}" for i in data_qubits]
    idle_gates = [layout_c.param(gate_label, d)["local"] for d in data_qubits]
    assert idle_gates == ["I" for _ in data_qubits]

    data_qubits = layout_t.get_qubits(role="data")
    idle_gates = [layout_t.param(gate_label, d)["local"] for d in data_qubits]
    assert idle_gates == ["H" for _ in data_qubits]

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

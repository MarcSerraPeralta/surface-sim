from surface_sim.log_gates.util import set_x, set_z, set_idle
from surface_sim.layouts import unrot_surface_code
from surface_sim.detectors import get_new_stab_dict_from_layout


def test_set_x():
    layout = unrot_surface_code(distance=3)
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
    layout = unrot_surface_code(distance=3)
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
    layout = unrot_surface_code(distance=3)
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

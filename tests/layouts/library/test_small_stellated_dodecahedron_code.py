from surface_sim.layouts.operations import check_code_definition
from surface_sim.layouts import ssd_code


def test_get_layout():
    layout = ssd_code()

    check_code_definition(layout)

    assert layout.num_logical_qubits == 8
    assert layout.num_data_qubits == 30
    assert layout.num_anc_qubits == 24

    return

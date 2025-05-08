from surface_sim.layouts.operations import check_code_definition
from surface_sim.layouts import ssd_code
from surface_sim.layouts.library.small_stellated_dodecahedron_code import (
    INTERACTION_ORDERS,
)


def test_ssd_code():
    layout = ssd_code()

    check_code_definition(layout)

    assert layout.num_logical_qubits == 8
    assert layout.num_data_qubits == 30
    assert layout.num_anc_qubits == 24

    return


def test_interaction_orders():
    layout = ssd_code()
    for int_order in INTERACTION_ORDERS.values():
        assert set(layout.anc_qubits) == set(int_order)
        data_qubits_steps = [[] for _ in int_order[layout.anc_qubits[0]]]

        for anc, steps in int_order.items():
            support = layout.get_neighbors([anc])

            data_qubits = [q for q in steps if q is not None]
            assert set(support) == set(data_qubits)

            for k, q in enumerate(steps):
                data_qubits_steps[k].append(q)

        for data_qubits_step in data_qubits_steps:
            # each data qubit is not interacting more than once per step
            data_qubits_step = [q for q in data_qubits_step if q is not None]
            assert len(data_qubits_step) == len(set(data_qubits_step))

    return

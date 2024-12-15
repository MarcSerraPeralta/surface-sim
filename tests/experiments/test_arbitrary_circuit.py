import stim

from surface_sim import Layout
from surface_sim.experiments import schedule_from_circuit
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes


def test_schedule_from_circuit():
    layouts = unrot_surface_codes(4, distance=3)
    circuit = stim.Circuit(
        """
        R 0 1 2
        TICK
        X 0
        M 1
        TICK
        CX 0 1 2 3
        """
    )

    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)

    list_num_ops = [3, 1, 2, 1, 2]
    for ops, num_ops in zip(schedule, list_num_ops):
        assert len(ops) == num_ops
        if num_ops == 1:
            assert ops[0][0].log_op_type == "qec_cycle"

    list_num_layouts = [[1, 1, 1], [0], [1, 1], [0], [2, 2]]
    for ops, num_layouts in zip(schedule, list_num_layouts):
        for op, num_l in zip(ops, num_layouts):
            assert len(op) == num_l + 1
            assert all(isinstance(l, Layout) for l in op[1:])
            if num_l != 0:
                assert op[0].log_op_type != "qec_cycle"

    return

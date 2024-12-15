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

    assert len(schedule) == 9

    list_num_layouts = [1, 1, 1, 0, 1, 1, 0, 2, 2]
    for op, num_layouts in zip(schedule, list_num_layouts):
        assert len(op) == num_layouts + 1
        assert all(isinstance(l, Layout) for l in op[1:])
        if num_layouts != 0:
            assert op[0].log_op_type != "qec_cycle"

    return

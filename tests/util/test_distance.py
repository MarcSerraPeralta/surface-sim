import stim

from surface_sim.util import get_circuit_distance

def test_get_circuit_distance():
    circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.01)
    d_circ = get_circuit_distance(circuit)
    assert d_circ == 3
    return

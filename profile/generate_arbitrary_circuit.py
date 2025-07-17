import stim

from surface_sim.setup import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

# INPUTS
DISTANCE = 41
PROB = 1e-3
NOISE_MODEL = CircuitNoiseModel
FRAME = "pre-gate"

CIRCUIT = stim.Circuit(
    """
    R 0 1
    TICK
    CNOT 0 1
    TICK
    S 0
    I 1
    TICK
    H 0 1
    TICK
    H 0 1
    TICK
    S 0
    I 1
    TICK
    CNOT 0 1
    TICK
    M 0 1
    """
)

layouts = unrot_surface_codes(CIRCUIT.num_qubits, distance=DISTANCE)

setup = CircuitNoiseSetup()
setup.set_var_param("prob", PROB)
model = NOISE_MODEL.from_layouts(setup=setup, *layouts)
detectors = Detectors.from_layouts(FRAME, *layouts)

schedule = schedule_from_circuit(CIRCUIT, layouts, gate_to_iterator)
experiment = experiment_from_schedule(
    schedule, model, detectors, anc_reset=True, anc_detectors=None
)

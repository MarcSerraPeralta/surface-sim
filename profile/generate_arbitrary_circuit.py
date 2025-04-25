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
qubit_inds = {}
anc_coords = {}
anc_qubits = []
stab_coords = {}
for l, layout in enumerate(layouts):
    qubit_inds.update(layout.qubit_inds())
    anc_qubits += layout.anc_qubits
    coords = layout.anc_coords
    anc_coords.update(coords)
    stab_coords[f"Z{l}"] = [v for k, v in coords.items() if k[0] == "Z"]
    stab_coords[f"X{l}"] = [v for k, v in coords.items() if k[0] == "X"]

setup = CircuitNoiseSetup()
setup.set_var_param("prob", PROB)
model = NOISE_MODEL(setup=setup, qubit_inds=qubit_inds)
detectors = Detectors(anc_qubits, frame=FRAME, anc_coords=anc_coords)

schedule = schedule_from_circuit(CIRCUIT, layouts, gate_to_iterator)
experiment = experiment_from_schedule(
    schedule, model, detectors, anc_reset=True, anc_detectors=None
)

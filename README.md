# Surface-sim

![example workflow](https://github.com/MarcSerraPeralta/surface-sim/actions/workflows/ci_pipeline.yaml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/surface-sim?label=pypi%20package)


This package is a wrapper around Stim to simplify the construction of QEC circuits.
Given a circuit, it can implement the logical equivalent under different types of noise,
including circuit-level noise.
It uses a code layout that helps with qubit labeling, indexing and connectivity. 
It also defines the detectors automatically for any sequence of logical gates.

For more information see the documentation in `docs/`. 

## Installation

This package is available in PyPI, thus it can be installed using
```
pip install surface-sim
```

or alternatively, it can be installed from source using
```
git clone git@github.com:MarcSerraPeralta/surface-sim.git
pip install surface-sim/
```

## Example

### Pre-built experiment: memory experiment

```
from surface_sim.layouts import rot_surface_code
from surface_sim.models import CircuitNoiseModel
from surface_sim.setup import CircuitNoiseSetup
from surface_sim import Detectors
from surface_sim.experiments.rot_surface_code_css import memory_experiment

# prepare the layout, model, and detectors objects
layout = rot_surface_code(distance=3)
setup = CircuitNoiseSetup()
model = CircuitNoiseModel(setup, layout.qubit_inds)
detectors = Detectors(layout.anc_qubits, frame="pre-gate")

# create a memory experiment
NUM_ROUNDS = 10
DATA_INIT = {q: 0 for q in layout.data_qubits}
ROT_BASIS = True # X basis
MEAS_RESET = True # reset after ancilla measurements
PROB = 1e-5

setup.set_var_param("prob", PROB)
stim_circuit = memory_experiment(model, layout, detectors, NUM_ROUNDS, DATA_INIT, ROT_BASIS, MEAS_RESET)
```

### Arbitrary logical circuit from a given circuit

```
import stim

from surface_sim.setup import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes

circuit = stim.Circuit(
    """
    R 0 1
    TICK
    CNOT 0 1
    TICK
    S 0
    I 1
    TICK
    S 0
    H 1
    TICK
    M 0
    MX 1
    """
)

layouts = unrot_surface_codes(circuit.num_qubits, distance=3)
setup = CircuitNoiseSetup()
model = CircuitNoiseModel.from_layouts(setup, *layouts)
detectors = Detectors.from_layouts("pre-gate", *layouts)

setup.set_var_param("prob", 1e-3)

schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
stim_circuit = experiment_from_schedule(
    schedule, model, detectors, anc_reset=True
)
```

For more information and examples about `surface-sim`, please read the `docs/`.

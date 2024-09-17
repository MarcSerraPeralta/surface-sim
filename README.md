# Surface-sim

![example workflow](https://github.com/MarcSerraPeralta/surface-sim/actions/workflows/actions.yaml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/surface-sim?label=pypi%20package)


This package is a wrapper around Stim that aims to help the construction of QEC-code circuits easier. The package implements model classes that help with inserting ciruit-level noise models. It uses a code layout that helps with qubit labeling, indexing and connectivity. Finally, there are a number of circuits for the rotated surface code that are implemented in the package.

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

```
from surface_sim.layouts import rot_surf_code
from surface_sim.models import CircuitNoiseModel
from surface_sim.setup import CircuitNoiseSetup
from surface_sim import Detectors
from surface_sim.experiments.surface_code_css import memory_experiment

# prepare the layout, model, and detectors objects
layout = rot_surf_code(distance=3)

qubit_inds = {q: layout.get_inds([q])[0] for q in layout.get_qubits()}
anc_qubits = layout.get_qubits(role="anc")
data_qubits = layout.get_qubits(role="data")

setup = CircuitNoiseSetup()

model = CircuitNoiseModel(setup, qubit_inds)

detectors = Detectors(anc_qubits, frame="1")

# create a memory experiment
NUM_ROUNDS = 10
DATA_INIT = {q: 0 for q in data_qubits}
ROT_BASIS = True # X basis
MEAS_RESET = True # reset after ancilla measurements
PROB = 1e-5

setup.set_var_param("prob", PROB)
stim_circuit = memory_experiment(model, layout, detectors, NUM_ROUNDS, DATA_INIT, ROT_BASIS, MEAS_RESET)
```

For more information and examples about `surface-sim`, please read the `docs/`.

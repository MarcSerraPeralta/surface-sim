# Creating and configuring a `Setup`

This file contains information on how to create and configure a `Setup` object for a noise model `Model` using (1) a YAML file, and (2) a `dict` object. 
This document explains the structure of these two inputs and the use of *free parameters* and *global parameters*. 

For convenience, `surface_sim.setup.CircuitNoiseSetup` returns a `Setup` initialized for circuit-level noise.


## Structure of the configuration for a `Setup`

The configuration input must contain a `setup` block, including the noise parameters for each qubit.
It can also contain a `gate_durations` block for experimental-based noise models, where the error probabilities depend on the opertaions durations. 

A common set of noise parameters for each qubits are:
- `sq_error_prob`: float,
- `cz_error_prob`: float,
- `meas_error_prob`: float, # quantum measurement errors
- `assign_error_prob`: float, # classical measurement error
- `reset_error_prob`: float,
- `idle_error_prob`: float,
- `T1`: float,
- `T2`: float,

*Note: not all parameters are required for each noise model*

*Note: the units of T1, T2 and gate durations must match*

The configuration input can also contain a `name` and a `description`. 

The parameters are classified into three categories:

- **local parameter**: float value that is defined for a specific qubit or pair of qubits for the case of two-qubit gates
- **global parameter**: float value that is defined for all qubits or all two-qubit gates
- **free parameter**: string name that can be set up and modified for an specific qubit or all qubits. 
These parameters can be setup using the `Setup.set_var_param` function. 

Examples that represent the same noise:
```
# local parameters
setup_input = [
    {
        "qubit": "D1", 
        "sq_error_prob": 0.001 
    },
    {
        "qubit": "D2", 
        "sq_error_prob": 0.001 
    },
    {
        "qubits": ["D1", "D2"], 
        "cz_error_prob": 0.01
    },
]
```

```
# global parameters
setup_input = [
    {
        "sq_error_prob": 0.001 
        "cz_error_prob": 0.01 
    },
]
```

```
# free parameters
setup_input = [
    {
        # global free parameter
        "sq_error_prob": "param1"
    },
    {
        # local free parameter
        "qubits": ["D1", "D2"], 
        "cz_error_prob": "param2" 
    },
]
# set the free parameters once the Setup object has been created:
# setup.set_var_param("param1", 0.001)
# setup.set_var_param("param2", 0.01)
```


## Loading `Setup` from YAML file

```
from surface_sim import Setup

setup = Setup.from_yaml("path/to/yaml/file.yaml")
```

Examples of the YAML Setup file can be found in `docs/setup_examples`.


## Creating `Setup` from `dict`

```
from surface_sim import Setup

setup_dict = {
    "gate_durations": {
        "X": 3.2,
        "Z": 1,
        "H": 1,
        "CZ": 1,
        "M": 1,
        "R": 1,
    },
    "setup": [
        {
            "cz_error_prob": 0.1,
            "meas_error_prob": 0.1,
            "assign_error_flag": True,
            "assign_error_prob": 0.1,
            "reset_error_prob": 0.1,
            "idle_error_prob": 0.1,
            "T1": 1,
            "T2": 1,
        },
        {
            "qubit": "D1", 
            "sq_error_prob": 0.001 
        }
        {
            "qubit": "D2", 
            "sq_error_prob": 0.002 
        }
    ],
    "name": "test",
    "description": "test description",
}

setup = Setup(setup_dict)
```

# Building noisy circuits with `Model`, `Layout`, and `Detectors` objects

The noise model class contains the functions that implement noisy operations and the layout class contains the possible interactions between qubits.
The detectors are managed automatically using the `Detectors` class. 
For an overview of the relations between these classes, see `docs/module_blocks_overview.svg`.

The `Model`'s methods for operations return a `stim.Circuit` corresponding to the specified operation and its associated noise. 
These instructions can be added to a `stim.Circuit`, for example:

```
# apply an X gate to qubit D1 and D2

circuit = stim.Circuit()
circuit += model.x_gate(["D1", "D2"]):
```

The building blocks in `surface_sim.circuit_blocks` use a layout (from `surface_sim.Layout`), which simplifies the qubit selection for gate scheduling. 
As an example, `Layout.get_qubits(role="anc")` selects all ancilla qubits from the layout.
In `docs/layout_examples`, there is a YAML file that stores the Surface-17 layout, which can be loaded using `Layout.from_yaml()`.
This package also contains functions to generate layouts in `surface_sim.layouts`.

To understand how to build noisy circuits, we suggest reading the code for the functions in `surface_sim.circuit_blocks.rot_surface_code_css`.
It is also important to know that every time a new circuit or experiment is build, the `Model` and `Detectors` classes need to be restarted with
```
model.new_circuit()
detectors.new_circuit()
```
This restarts the inner state of the classes, in particular for the function `Model.meas_target` and the stabilizers in `Detectors`.

Finally, this package already contains some common QEC experiments in `surface_sim.experiments`.

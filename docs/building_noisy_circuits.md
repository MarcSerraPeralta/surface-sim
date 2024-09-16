# Building noisy circuits with `Model` and `Layout` objects

The noise model class contains the functions that implement noisy operations and the layout class contains the possible interactions between qubits.
The `Model` gate functions yield `stim.CircuitInstruction`s corresponding to the specified gate and its associated noise. 
These instructions can be appended to a `stim.Circuit`, for example:

```
# apply an X gate to qubit D1 and D2

circuit = stim.Circuit()
for instr in model.x_gate(["D1", "D2"]):
    circuit.append(instr)
```

The building blocks in `surface_sim.circuit_blocks` use a layout (from `qec_util.Layout`), which simplifies the qubit selection for gate scheduling. 
As an example, `Layout.get_qubits(role="anc")` selects all ancilla qubits from the layout.
In `docs/layout_examples`, there is a YAML file that stores the Surface-17 layout and which can be loaded using `Layout.from_yaml()`.


# Example: QEC cycle for the standard pipelined surface code

Below, the code for generating the QEC cycle for the standard pipelined surface code is explained. 

Firstly, we define some useful variables for applying gates to specific types of qubits, i.e. data qubits and ancilla qubits.

*Note: when not reseting the ancillas after the measurement in the QEC cycle, 
the definition of the defects as a function of the ancilla outcomes is different than when using reset.*

```
data_qubits = layout.get_qubits(role="data")
anc_qubits = layout.get_qubits(role="anc")

qubits = set(data_qubits + anc_qubits)

# With reset defect[n] = m[n] XOR m[n-1]
# Wihtout reset defect[n] = m[n] XOR m[n-2]
comp_round = 1 if meas_reset else 2
```

The gate scheduling of the gates follows a specific pattern.
For the two-qubit gates, such a pattern is known as the CZ dance and it is stored in the layout object as interaction order.
This order depends on the stabilizer type, i.e. "x_type" or "z_type". 

```
circuit = Circuit()
int_order = layout.interaction_order
stab_types = list(int_order.keys())
```

To measure the "x_type" stabilizers, we need to perform a Hadamard gate to the ancilla and data qubits in the first step of the dance. 

*Note: it is practical to keep track of the qubits that have been driven to add idling to the rest of the qubits.*

```
for ind, stab_type in enumerate(stab_types):
    stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
    rot_qubits = set(stab_qubits)
    if stab_type == "x_type":
        rot_qubits.update(data_qubits)

    if not ind:
        for instruction in model.hadamard(rot_qubits):
            circuit.append(instruction)

        idle_qubits = qubits - rot_qubits
        for instruction in model.idle(idle_qubits):
            circuit.append(instruction)
        circuit.append("TICK")
```

We loop over the CZ dance steps. 
The pairs of qubits are obtained from searching the neighbouring qubits in the direction `ord_dir` (from NE, NW, SE, SW)
of the ancilla qubits that measure the `stab_type` stabilizers (from "x_type" and "z_type"). 

```
    for ord_dir in int_order[stab_type]:
        int_pairs = layout.get_neighbors(
            stab_qubits, direction=ord_dir, as_pairs=True
        )
        int_qubits = list(chain.from_iterable(int_pairs))

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

        idle_qubits = qubits - set(int_qubits)
        for instruction in model.idle(idle_qubits):
            circuit.append(instruction)
        circuit.append("TICK")
```

If we still need to measure the other type of stabilizers, we perform Hadamard gates to all ancilla and data qubits. 
If not, we just undo the rotation to the qubits to which we have applied a Hadamard gate. 

```
    if not ind:
        for instruction in model.hadamard(qubits):
            circuit.append(instruction)
    else:
        for instruction in model.hadamard(rot_qubits):
            circuit.append(instruction)

        idle_qubits = qubits - rot_qubits
        for instruction in model.idle(idle_qubits):
            circuit.append(instruction)

    circuit.append("TICK")
```

Then, we measure the ancilla qubits to obtain the stabilizer outcomes, and reset the ancillas if necessary. 
```
for instruction in model.measure(anc_qubits):
    circuit.append(instruction)

for instruction in model.idle(data_qubits):
    circuit.append(instruction)
circuit.append("TICK")

if meas_reset:
    for instruction in model.reset(anc_qubits):
        circuit.append(instruction)
    for instruction in model.idle(data_qubits):
        circuit.append(instruction)

    circuit.append("TICK")
```

Finally, we define the detectors for the QEC cycle. 
The `meas_comparison` flag is used to differenciate the defect formulas for the first rounds and for the bulk,
which are different, i.e. *d[1] = m[1]* while *d[n] = m[n] ^ m[n-k]* where *k* depends if the ancillas are reset or not.

*Note: for more information on the defects/detector definitions, see Stim's documentation.*

```
# detectors ordered as in the measurements
num_anc = len(anc_qubits)
if meas_comparison:
    det_targets = []
    for ind in range(num_anc):
        target_inds = [ind - (comp_round + 1) * num_anc, ind - num_anc]
        targets = [target_rec(ind) for ind in target_inds]
        det_targets.append(targets)
else:
    det_targets = [[target_rec(ind - num_anc)] for ind in range(num_anc)]

for targets in det_targets:
    circuit.append("DETECTOR", targets)
```

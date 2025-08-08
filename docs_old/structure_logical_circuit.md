# Structure of a logical circuit

The structure that supported in `surface_sim` is the following
```
logical operation layer | QEC round | logical operation layer | QEC round | ...
```

where in a logical operation layer each logical qubit MUST performs ONE logical operation.
A logical operation is a logical reset, measurement, or unitary gate.
Logical idling is considered an operaion. 
A circuit can never have a logical reset followed directly by a logical measurement. 
There will be always a QEC round between them. 

It is possible to perform more than one QEC round between logical operation layers:
```
logical operation layer | QEC round | ... | QEC round| logical operation layer | QEC round | ...
```

The logical operations must have one `model.tick()` before starting the physical implementation of the operation.
This TICK is used to add incoming noise (if needed) and to plot the stabilizers before the gate when calling `stim.Circuit.diagram`.
The TICK must be there to make sure that idling noise is correctly added when the logical operations have different physical gate layers.

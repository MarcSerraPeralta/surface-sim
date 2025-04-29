# Structure of a logical circuit

The structure that supported in `surface_sim` is the following
```
logical operation layer | QEC cycle | logical operation layer | QEC cycle | ...
```

where in a logical operation layer each logical qubit MUST performs ONE logical operation.
A logical operation is a logical reset, measurement, or unitary gate.
Logical idling is considered an operaion. 
A circuit can never have a logical reset followed directly by a logical measurement. 
There will be always a QEC cycle between them. 

It is possible to perform more than one QEC cycle between logical operation layers:
```
logical operation layer | QEC cycle | ... | QEC cycle| logical operation layer | QEC cycle | ...
```

from typing import Any, Dict, Sequence, List

from stim import CircuitInstruction

from ..setup import Setup


class Model(object):
    def __init__(self, setup: Setup, qubit_inds: Dict[str, int]) -> None:
        self._setup = setup
        self._qubit_inds = qubit_inds

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def qubits(self) -> List[str]:
        return list(self._qubit_inds.keys())

    def gate_duration(self, name: str) -> float:
        return self._setup.gate_duration(name)

    def get_inds(self, qubits: Sequence[str]) -> List[int]:
        return [self._qubit_inds[q] for q in qubits]

    def param(self, *qubits: str) -> Any:
        return self._setup.param(*qubits)

    # annotation gates
    def tick(self, **_):
        yield CircuitInstruction("TICK", targets=[])

    def qubit_coords(self, coords: Dict[str, list]):
        if set(coords) < set(self._qubit_inds):
            raise ValueError("'coords' must have the coordinates for all the qubit")

        for q_label, q_ind in self._qubit_inds.items():
            q_coords = coords[q_label]
            yield CircuitInstruction("QUBIT_COORDS", [q_ind], q_coords)

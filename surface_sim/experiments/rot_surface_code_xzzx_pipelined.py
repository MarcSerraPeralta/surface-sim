from copy import deepcopy
from stim import Circuit

from ..circuit_blocks.rot_surface_code_xzzx_pipelined import (
    init_qubits_iterator,
    gate_to_iterator,
)
from .arbitrary_experiment import experiment_from_schedule, schedule_from_circuit
from ..layouts.layout import Layout
from ..models import Model
from ..detectors import Detectors
from ..circuit_blocks.decorators import (
    qubit_init_z,
    qubit_init_x,
)


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = True,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a memory experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_rounds
        Number of QEC cycle to run in the memory experiment.
    data_init
        Bitstring for initializing the data qubits.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If ``True``, ancillas are reset at the beginning of the QEC cycle.
        By default ``True``.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(
            f"'num_rounds' expected as int, got {type(num_rounds)} instead."
        )
    if num_rounds < 0:
        raise ValueError("'num_rounds' needs to be a positive integer.")
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    reset = qubit_init_x if rot_basis else qubit_init_z

    @reset
    def custom_reset_iterator(m: Model, l: Layout):
        return init_qubits_iterator(m, l, data_init=data_init, rot_basis=rot_basis)

    b = "X" if rot_basis else "Z"
    custom_gate_to_iterator = deepcopy(gate_to_iterator)
    custom_gate_to_iterator[f"R{b}"] = custom_reset_iterator

    unencoded_circuit = Circuit(f"R{b} 0" + "\nTICK" * num_rounds + f"\nM{b} 0")
    schedule = schedule_from_circuit(
        unencoded_circuit, layouts=[layout], gate_to_iterator=custom_gate_to_iterator
    )
    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=anc_reset, anc_detectors=anc_detectors
    )

    return experiment

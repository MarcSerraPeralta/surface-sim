import warnings

from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.rot_surface_code_css import (
    init_qubits,
    log_meas,
    qec_round,
    qubit_coords,
    log_trans_s,
)
from ..models import Model
from ..detectors import Detectors


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = False,
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
        If True, ancillas are reset at the beginning of the QEC cycle.
        By default True.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds < 0:
        raise ValueError("num_rounds needs to be a positive integer.")
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)

    for _ in range(num_rounds):
        experiment += qec_round(model, layout, detectors, anc_reset, anc_detectors)
    experiment += log_meas(
        model, layout, detectors, rot_basis, anc_reset, anc_detectors
    )

    return experiment


def repeated_s_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_s_gates: int,
    num_rounds_per_gate: int,
    data_init: dict[str, int] | list[int],
    rot_basis: bool = False,
    anc_reset: bool = False,
    anc_detectors: list[str] | None = None,
) -> Circuit:
    """Returns the circuit for running a repeated-S experiment.

    Parameters
    ----------
    model
        Noise model for the gates.
    layout
        Code layout.
    detectors
        Detector definitions to use.
    num_s_gates
        Number of logical (transversal) S gates to run in the experiment.
    num_rounds_per_gate
        Number of QEC cycles to be run after each logical S gate.
    data_init
        Bitstring for initializing the data qubits.
    rot_basis
        If ``True``, the memory experiment is performed in the X basis.
        If ``False``, the memory experiment is performed in the Z basis.
        By deafult ``False``.
    anc_reset
        If True, ancillas are reset at the beginning of the QEC cycle.
        By default True.
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds_per_gate, int):
        raise ValueError(
            f"num_rounds_per_gate expected as int, got {type(num_rounds_per_gate)} instead."
        )
    if num_rounds_per_gate < 0:
        raise ValueError("num_rounds_per_gate needs to be a positive integer.")

    if not isinstance(num_s_gates, int):
        raise ValueError(
            f"num_s_gates expected as int, got {type(num_s_gates)} instead."
        )
    if (num_s_gates < 0) or (num_s_gates % 2 == 1):
        raise ValueError("num_s_gates needs to be an even positive integer.")

    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)
    experiment += qec_round(model, layout, detectors, anc_reset, anc_detectors)

    for _ in range(num_s_gates):
        experiment += log_trans_s(model, layout, detectors)
        for _ in range(num_rounds_per_gate):
            experiment += qec_round(model, layout, detectors, anc_reset, anc_detectors)
    experiment += log_meas(
        model, layout, detectors, rot_basis, anc_reset, anc_detectors
    )

    return experiment

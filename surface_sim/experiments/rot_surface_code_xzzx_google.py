import warnings

from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.rot_surface_code_xzzx_google import (
    init_qubits,
    qec_round_with_log_meas,
    qec_round,
    qubit_coords,
)
from ..models import Model
from ..detectors import Detectors


def memory_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | list[int],
    anc_detectors: list[str] | None = None,
    rot_basis: bool = False,
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
    anc_detectors
        List of ancilla qubits for which to define the detectors.
        If ``None``, adds all detectors.
        By default ``None``.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a (strickly) positive integer.")
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)

    if num_rounds == 1:
        experiment += qec_round_with_log_meas(
            model, layout, detectors, anc_detectors, rot_basis
        )
        return experiment

    for _ in range(num_rounds - 1):
        experiment += qec_round(model, layout, detectors, anc_detectors)
    experiment += qec_round_with_log_meas(
        model, layout, detectors, anc_detectors, rot_basis
    )

    return experiment

from typing import List, Dict
import warnings

from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.surface_code_css_pipelined import (
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
    data_init: Dict[str, int] | List[int],
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds < 0:
        raise ValueError("num_rounds needs to be a positive integer.")
    if isinstance(data_init, list) and len(set(data_init)) == 1:
        data_init = {q: data_init[0] for q in layout.get_qubits(role="data")}
        warnings.warn("'data_init' should be a dict.", DeprecationWarning)
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    model.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)

    for _ in range(num_rounds):
        experiment += qec_round(model, layout, detectors, meas_reset)
    experiment += log_meas(model, layout, detectors, rot_basis, meas_reset)

    return experiment


def repeated_s_experiment(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_s_gates: int,
    num_rounds_per_gate: int,
    data_init: Dict[str, int] | List[int],
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
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

    if isinstance(data_init, list) and len(set(data_init)) == 1:
        data_init = {q: data_init[0] for q in layout.get_qubits(role="data")}
        warnings.warn("'data_init' should be a dict.", DeprecationWarning)
    if not isinstance(data_init, dict):
        raise TypeError(f"'data_init' must be a dict, but {type(data_init)} was given.")

    model.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)
    experiment += qec_round(model, layout, detectors, meas_reset)

    for _ in range(num_s_gates):
        experiment += log_trans_s(model, layout, detectors)
        for _ in range(num_rounds_per_gate):
            experiment += qec_round(model, layout, detectors, meas_reset)
    experiment += log_meas(model, layout, detectors, rot_basis, meas_reset)

    return experiment

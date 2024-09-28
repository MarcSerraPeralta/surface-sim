import warnings

from stim import Circuit

from ..layouts import Layout
from ..circuit_blocks.surface_code_xzzx import (
    init_qubits,
    log_meas,
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
    rot_basis: bool = False,
    anc_reset: bool = False,
    meas_reset: bool | None = None,
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
    if meas_reset is not None:
        warnings.warn("Use 'anc_reset' instead of 'meas_reset'", DeprecationWarning)
        anc_reset = meas_reset

    model.new_circuit()
    detectors.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)

    for _ in range(num_rounds):
        experiment += qec_round(model, layout, detectors, anc_reset)
    experiment += log_meas(model, layout, detectors, rot_basis, anc_reset)

    return experiment

from typing import List

from stim import Circuit

from qec_util import Layout

from ..circuit_blocks.surface_code_xzzx_google import (
    init_qubits,
    qec_round_with_log_meas,
    qec_round,
    qubit_coords,
)
from ..models import Model


def memory_experiment(
    model: Model,
    layout: Layout,
    num_rounds: int,
    data_init: List[int],
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")

    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a (strickly) positive integer.")

    num_init_rounds = 1 if meas_reset else 2
    model.new_circuit()

    experiment = Circuit()
    experiment += qubit_coords(model, layout)
    experiment += init_qubits(model, layout, data_init, rot_basis)

    if num_rounds <= num_init_rounds:
        for _ in range(min(num_rounds, num_init_rounds) - 1):
            experiment += qec_round(model, layout, meas_reset, meas_comparison=False)
        experiment += qec_round_with_log_meas(
            model, layout, rot_basis, meas_reset=False
        )
        return experiment
    else:
        for _ in range(num_init_rounds):
            experiment += qec_round(model, layout, meas_reset, meas_comparison=False)
        for _ in range(num_rounds - num_init_rounds - 1):
            experiment += qec_round(model, layout, meas_reset)
        experiment += qec_round_with_log_meas(model, layout, rot_basis, meas_reset)
        return experiment

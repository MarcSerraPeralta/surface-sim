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

    qubit_coords_circ = qubit_coords(model, layout)
    init_circ = init_qubits(model, layout, data_init, rot_basis)
    qec_meas_circuit = qec_round_with_log_meas(model, layout, rot_basis, meas_reset)
    first_qec_circ = qec_round(model, layout, meas_reset, meas_comparison=False)

    if num_rounds > num_init_rounds:
        qec_circ = qec_round(model, layout, meas_reset)

        experiment = (
            qubit_coords_circ
            + init_circ
            + first_qec_circ * num_init_rounds
            + qec_circ * (num_rounds - 1 - num_init_rounds)
            + qec_meas_circuit
        )

        return experiment

    experiment = (
        qubit_coords_circ
        + init_circ
        + first_qec_circ * (min(num_rounds, num_init_rounds) - 1)
        + qec_round_with_log_meas(
            model, layout, rot_basis, meas_reset=True, meas_comparison=False
        )
    )

    return experiment

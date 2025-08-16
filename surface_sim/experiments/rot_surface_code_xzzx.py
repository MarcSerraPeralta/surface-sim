from stim import Circuit

from .arbitrary_experiment import experiment_from_schedule
from ..circuit_blocks.rot_surface_code_xzzx import (
    gate_to_iterator,
    init_qubits_iterator,
    init_qubits_z0_iterator,
    init_qubits_x0_iterator,
    qec_round_google_iterator,
    qec_round_google_with_log_x_meas_iterator,
    qec_round_google_with_log_z_meas_iterator,
)
from ..circuit_blocks.decorators import LogOpCallable, qubit_init_z, qubit_init_x
from . import templates
from ..layouts import Layout
from ..models import Model
from ..detectors import Detectors
from .arbitrary_experiment import Schedule


def memory_experiment(
    *args,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    **kargs,
) -> Circuit:
    """For information, see ``surface_sim.experiments.templates.memory_experiment``."""
    return templates.memory_experiment(
        *args,
        gate_to_iterator=gate_to_iterator,
        init_qubits_iterator=init_qubits_iterator,
        **kargs,
    )


def memory_experiment_google(
    model: Model,
    layout: Layout,
    detectors: Detectors,
    num_rounds: int,
    data_init: dict[str, int] | None = None,
    rot_basis: bool = False,
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
        Number of QEC round to run in the memory experiment.
    data_init
        Bitstring for initializing the data qubits. By default ``None`` mearning
        that it initializes the qubits using the reset given by ``gate_to_iterator``.
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
        raise ValueError(
            f"'num_rounds' expected as int, got {type(num_rounds)} instead."
        )
    if num_rounds <= 0:
        raise ValueError("'num_rounds' needs to be a strictly positive integer.")

    custom_reset_iterator = (
        init_qubits_x0_iterator if rot_basis else init_qubits_z0_iterator
    )
    if data_init is not None:
        reset = qubit_init_x if rot_basis else qubit_init_z

        @reset
        def custom_reset_iterator(model: Model, layout: Layout):
            return init_qubits_iterator(
                model, layout, data_init=data_init, rot_basis=rot_basis
            )

    schedule: Schedule = [[(custom_reset_iterator, layout)]]
    for _ in range(num_rounds - 1):
        schedule.append([(qec_round_google_iterator, layout)])
    if rot_basis:
        schedule.append([(qec_round_google_with_log_x_meas_iterator, layout)])
    else:
        schedule.append([(qec_round_google_with_log_z_meas_iterator, layout)])

    experiment = experiment_from_schedule(
        schedule, model, detectors, anc_reset=True, anc_detectors=anc_detectors
    )

    return experiment


def stability_experiment(
    *args,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    **kargs,
) -> Circuit:
    """For information, see ``surface_sim.experiments.templates.stability_experiment``."""
    return templates.stability_experiment(
        *args,
        gate_to_iterator=gate_to_iterator,
        init_qubits_iterator=init_qubits_iterator,
        **kargs,
    )

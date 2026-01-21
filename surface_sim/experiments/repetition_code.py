from stim import Circuit

from ..circuit_blocks.repetition_code import gate_to_iterator, init_qubits_iterator
from ..circuit_blocks.decorators import LogOpCallable
from . import templates


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

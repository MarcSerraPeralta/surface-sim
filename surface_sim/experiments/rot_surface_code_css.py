from stim import Circuit

from ..circuit_blocks.rot_surface_code_css import gate_to_iterator, init_qubits_iterator
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


def repeated_s_experiment(
    *args,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    **kargs,
) -> Circuit:
    """For information, see ``surface_sim.experiments.templates.repeated_s_experiment``."""
    return templates.repeated_s_experiment(
        *args,
        gate_to_iterator=gate_to_iterator,
        init_qubits_iterator=init_qubits_iterator,
        **kargs,
    )


def repeated_cnot_experiment(
    *args,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    **kargs,
) -> Circuit:
    """For information, see ``surface_sim.experiments.templates.repeated_cnot_experiment``."""
    return templates.repeated_cnot_experiment(
        *args,
        gate_to_iterator=gate_to_iterator,
        init_qubits_iterator=init_qubits_iterator,
        **kargs,
    )


def repeated_s_injection_experiment(
    *args,
    gate_to_iterator: dict[str, LogOpCallable] = gate_to_iterator,
    init_qubits_iterator: LogOpCallable | None = init_qubits_iterator,
    **kargs,
) -> Circuit:
    """For information, see ``surface_sim.experiments.templates.repeated_s_injection_experiment``."""
    return templates.repeated_s_injection_experiment(
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

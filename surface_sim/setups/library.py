from .setup import Setup


class CircuitNoiseSetup(Setup):
    def __init__(self) -> None:
        """
        Initialises a ``Setup`` class for the standard circuit-level noise.

        It contains a variable parameter ``"prob"`` that can be set for
        different physical error probabilities.
        """
        setup_dict = dict(
            name="Circuit-level noise setup",
            description="Setup for a circuit-level noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob}",
                    reset_error_prob="{prob}",
                    idle_error_prob="{prob}",
                    assign_error_flag=False,
                    assign_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class SD6(Setup):
    def __init__(self) -> None:
        """
        Initialises a ``Setup`` class for the SD6 noise described in:

        C. Gidney, M. Newman, A. Fowler, and M. Broughton.
        A Fault-Tolerant honeycomb memory. Quantum, 5:605, Dec. 2021.
        https://doi.org/10.22331/q-2022-09-21-813

        **IMPORTANT**

        1. It should be loaded with the ``SD6NoiseModel`` model. It should not be loaded
        with ``CircuitNoiseModel`` because the SD6 noise model does not support all
        Clifford gates.

        2. This noise model assumes that qubits are reset after measurements.
        In this sense, it does not add classical measurement errors (also known as
        assignment errors). It also assumes that ``model.tick()`` is called
        in-between gate layers.

        3. It contains a variable parameter ``"prob"`` that must be set before
        building any circuit.
        """
        setup_dict = dict(
            name="SD6 noise setup",
            description="Setup for a SD6 noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob}",
                    reset_error_prob="{prob}",
                    idle_error_prob="{prob}",
                    assign_error_flag=False,
                    assign_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class UniformDepolarizing(Setup):
    def __init__(self) -> None:
        """
        Initialises a ``Setup`` class for the UniformDepolarizing noise described in:

        McEwen, M., Bacon, D., & Gidney, C. (2023).
        Relaxing hardware requirements for surface code circuits using time-dynamics. Quantum, 7, 1172.
        https://doi.org/10.22331/q-2023-11-07-1172

        **IMPORTANT**

        1. It should be loaded with the ``UniformDepolarizingNoiseModel`` model.
        It should not be loaded with ``CircuitNoiseModel`` because the
        UniformDepolarizing noise model does not support all Clifford gates.

        2. It contains a variable parameter ``"prob"`` that must be set before
        building any circuit.
        """
        setup_dict = dict(
            name="UniformDepolarizing noise setup",
            description="Setup for a UniformDepolarizing noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob}",
                    reset_error_prob="{prob}",
                    idle_error_prob="{prob}",
                    assign_error_flag=True,
                    assign_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class SI1000(Setup):
    def __init__(self) -> None:
        """
        Initialises a ``Setup`` class for the SI1000 noise described in:

        C. Gidney, M. Newman, A. Fowler, and M. Broughton.
        A Fault-Tolerant honeycomb memory. Quantum, 5:605, Dec. 2021.
        https://doi.org/10.22331/q-2022-09-21-813

        **IMPORTANT**

        1. It should be loaded with the ``SI1000NoiseModel`` model. It should not be loaded
        with ``CircuitNoiseModel`` because the noise model stacks noise channels
        for qubits that are not being measured on top of their respective
        noise gate channels (e.g. idling).

        2. This noise model assumes that qubits are reset after measurements.
        In this sense, it does not add classical measurement errors (also known as
        assignment errors). It also assumes that ``model.tick()`` is called
        in-between gate layers.

        3. It contains a variable parameter ``"prob"`` that must be set before
        building any circuit.
        """
        setup_dict = dict(
            name="SI1000 noise setup",
            description="Setup for the SI1000 noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob} / 10",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob} * 5",
                    reset_error_prob="{prob} * 2",
                    idle_error_prob="{prob} / 10",
                    extra_idle_meas_or_reset_error_prob="{prob} * 2",
                    assign_error_flag=False,
                    assign_error_prob=0,
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class ExtendedSI1000(Setup):
    def __init__(self) -> None:
        """
        Initialises a ``Setup`` class for the ExtendedSI1000 noise described in:

        McEwen, M., Bacon, D., & Gidney, C. (2023).
        Relaxing hardware requirements for surface code circuits using time-dynamics. Quantum, 7, 1172.
        https://doi.org/10.22331/q-2023-11-07-1172

        **IMPORTANT**

        1. It should be loaded with the ``ExtendedSI1000NoiseModel`` model. It should not be loaded
        with ``CircuitNoiseModel`` because the noise model stacks noise channels
        for qubits that are not being measured on top of their respective
        noise gate channels (e.g. idling).

        2. This noise model assumes that ``model.tick()`` is called
        in-between gate layers.

        3. It contains a variable parameter ``"prob"`` that must be set before
        building any circuit.
        """
        setup_dict = dict(
            name="ExtendedSI1000 noise setup",
            description="Setup for the ExtendedSI1000 noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob} / 10",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob}",
                    reset_error_prob="{prob} * 2",
                    idle_error_prob="{prob} / 10",
                    extra_idle_meas_or_reset_error_prob="{prob} * 2",
                    assign_error_flag=True,
                    assign_error_prob="{prob} * 5",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class BiasedCircuitNoiseSetup(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for the biased circuit-level noise."""
        setup_dict = dict(
            name="Biased circuit-level noise setup",
            description="Setup for a Pauli-biased circuit-level noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob}",
                    reset_error_prob="{prob}",
                    idle_error_prob="{prob}",
                    assign_error_flag=False,
                    assign_error_prob="{prob}",
                    biased_pauli="{biased_pauli}",
                    biased_factor="{biased_factor}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class IncomingNoiseSetup(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for incoming noise."""
        setup_dict = dict(
            name="Incoming noise setup",
            description="Setup for incoming noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class PhenomenologicalNoiseSetup(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for phenomenological noise."""
        setup_dict = dict(
            name="Phenomenological noise setup",
            description="Setup for phenomenological noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    assign_error_flag=True,
                    assign_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class IncResMeasNoiseSetup(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for incoming+reset+measurement noise."""
        setup_dict = dict(
            name="Incoming+reset+measurement noise setup",
            description="Setup for incoming+reset+measurement noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    assign_error_flag=True,
                    assign_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class MeasurementNoiseSetup(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for measurement noise."""
        setup_dict = dict(
            name="Measurement noise setup",
            description="Setup for measurement noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob}",
                    assign_error_flag=True,
                    assign_error_prob="{prob}",
                ),
            ],
        )
        super().__init__(setup_dict)
        return


class NLR(Setup):
    def __init__(self) -> None:
        """
        Initialises a ``Setup`` class for the NLR noise described in:

        Beni, L. A., Higgott, O., & Shutty, N. (2025).
        Tesseract: A search-based decoder for quantum error correction.
        arXiv preprint arXiv:2503.10988.

        **IMPORTANT**

        1. It should be loaded with the ``NLRNoiseModel`` model. It should not be loaded
        with ``CircuitNoiseModel`` because the noise model stacks noise channels
        for qubits that are not being measured on top of their respective
        noise gate channels (e.g. idling).

        2. See other assumptions done in the ``SI1000`` setup.

        3. It contains three variable parameters that must be set before building any circuit:
        ``"prob"``, ``"long_coupler_distance"``, and ``"long_coupler_error_prob_factor"``.
        A coupler is considered long if the distance between the two qubits
        involved in the two-qubit gate is strictly larger than the specified
        ``long_coupler_distance``. For the long-range couplers, the noise
        strength is ``long_coupler_error_prob_factor`` times larger than
        standard two-qubit gates. In this sense, the ``NLR5`` noise model
        corresponds to ``long_coupler_error_prob_factor = 5``.
        """
        setup_dict = dict(
            name="NLR noise setup",
            description="Setup for the NLR noise model that can be used for any code and distance.",
            setup=[
                dict(
                    sq_error_prob="{prob} / 10",
                    tq_error_prob="{prob}",
                    long_range_tq_error_prob="{prob} * {long_coupler_error_prob_factor}",
                    long_coupler_distance="{long_coupler_distance}",
                    meas_error_prob="{prob} * 5",
                    reset_error_prob="{prob} * 2",
                    idle_error_prob="{prob} / 10",
                    extra_idle_meas_or_reset_error_prob="{prob} * 2",
                    assign_error_flag=False,
                    assign_error_prob=0,
                ),
            ],
        )
        super().__init__(setup_dict)
        return

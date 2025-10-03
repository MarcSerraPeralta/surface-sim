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

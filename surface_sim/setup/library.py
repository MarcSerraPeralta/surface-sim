from .setup import Setup


class CircuitNoiseSetup(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for circuit-level noise.

        It contains a variable parameter ``"prob"`` that can be set for
        different physical error probabilities.
        """
        setup_dict = dict(
            name="Circuit-level noise setup",
            description="Setup for a circuit-level noise model that can be used for any distance.",
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
        """Initialises a ``Setup`` class for the SI1000 circuit-level noise described in:
        C. Gidney, M. Newman, A. Fowler, and M. Broughton.
        A Fault-Tolerant honeycomb memory. Quantum, 5:605, Dec. 2021.

        It should be loaded with the `CircuitNoiseModel` model.

        It contains a variable parameter ``"prob"`` that can be set for
        different physical error probabilities.
        """
        setup_dict = dict(
            name="SI1000 noise setup",
            description="Setup for the SI1000 noise model that can be used for any distance.",
            setup=[
                dict(
                    sq_error_prob="{prob} / 10",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob} * 5",
                    reset_error_prob="{prob} * 2",
                    idle_error_prob="{prob} / 10",
                    idle_meas_error_prob="{prob} * 2",
                    idle_reset_error_prob="{prob} * 2",
                    assign_error_flag=False,
                    assign_error_prob=0,
                ),
            ],
        )
        super().__init__(setup_dict)
        return

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
                    sq_error_prob="prob",
                    tq_error_prob="prob",
                    meas_error_prob="prob",
                    reset_error_prob="prob",
                    idle_error_prob="prob",
                    assign_error_flag=True,
                    assign_error_prob="prob",
                ),
            ],
        )
        super().__init__(setup_dict)
        return

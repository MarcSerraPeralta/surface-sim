from __future__ import annotations
from collections.abc import Sequence
from typing import TypedDict
from copy import deepcopy
from pathlib import Path

import yaml

Param = float | int | bool | str | None


class SetupDict(TypedDict):
    setup: Sequence[dict[str, object]]


class Setup:
    PARENTS = {
        "cz_error_prob": "tq_error_prob",
        "cnot_error_prob": "tq_error_prob",
        "swap_error_prob": "tq_error_prob",
        "h_error_prob": "sq_error_prob",
        "s_error_prob": "sq_error_prob",
        "sdag_error_prob": "sq_error_prob",
        "x_error_prob": "sq_error_prob",
        "z_error_prob": "sq_error_prob",
        "reset_error_prob": "sq_error_prob",
        "meas_error_prob": "sq_error_prob",
        "idle_error_prob": "sq_error_prob",
        "idle_meas_error_prob": "idle_error_prob",
        "idle_reset_error_prob": "idle_error_prob",
    }

    def __init__(self, setup: SetupDict) -> None:
        """Initialises the ``Setup`` class.

        Parameters
        ----------
        setup
            Dictionary with the configuration.
            Must have the key ``"setup"`` containing the information.
            The information must not have ``None`` as value.
            It can also include ``"name"``, ``"description"`` and
            ``"gate_durations"`` keys with the corresponding information.
        """
        self._qubit_params = dict()
        self._global_params = dict()
        self._var_params = dict()
        self.uniform = False

        _setup = deepcopy(setup)
        self.name = _setup.pop("name", None)
        self.description = _setup.pop("description", None)
        self._gate_durations = _setup.pop("gate_durations", {})
        self._load_setup(_setup)
        if self._qubit_params == {}:
            self.uniform = True

        return

    def _load_setup(self, setup: SetupDict) -> None:
        params = setup.get("setup")
        if not params:
            raise ValueError("'setup['setup']' not found or contains no information.")

        for params_dict in params:
            if "qubit" in params_dict:
                qubit = str(params_dict.pop("qubit"))
                qubits = (qubit,)
            elif "qubits" in params_dict:
                qubits = tuple(params_dict.pop("qubits"))
            else:
                qubits = None

            if any(not isinstance(v, Param) for v in params_dict.values()):
                raise TypeError(f"Params must be {Param}, but {params_dict} was given.")

            if qubits:
                if qubits in self._qubit_params.keys():
                    raise ValueError("Parameters defined repeatedly in the setup.")
                self._qubit_params[qubits] = params_dict
            else:
                self._global_params.update(params_dict)

            for val in params_dict.values():
                if isinstance(val, str):
                    for p in _get_var_params(val):
                        self._var_params[p] = None

    @property
    def free_params(self) -> list[str]:
        """Returns the unset variable parameters."""
        return [param for param, val in self._var_params.items() if val is None]

    @classmethod
    def from_yaml(cls: type[Setup], filename: str | Path) -> Setup:
        """Create new ``surface_sim.setup.Setup`` instance from YAML
        configuarion file.

        Parameters
        ----------
        filename
            The YAML file name.

        Returns
        -------
        T
            The initialised ``surface_sim.setup.Setup`` object based on the yaml.
        """
        with open(filename, "r") as file:
            setup = yaml.safe_load(file)
            return cls(setup)

    def to_dict(self) -> SetupDict:
        """Returns a dictionary that can be used to initialize ``Setup``."""
        setup = dict()

        setup["name"] = self.name
        setup["description"] = self.description
        setup["gate_durations"] = self._gate_durations

        qubit_params = []
        if self._global_params:
            qubit_params.append(self._global_params)

        for qubits, params in self._qubit_params.items():
            params_copy = deepcopy(params)
            num_qubits = len(qubits)
            if num_qubits == 1:
                params_copy["qubit"] = qubits[0]
            elif num_qubits == 2:
                params_copy["qubits"] = list(qubits)
            qubit_params.append(params_copy)

        setup["setup"] = qubit_params

        return setup

    def to_yaml(self, filename: str | Path) -> None:
        """Stores the current ``Setup`` configuration in the given file
        in YAML format.

        Parameters
        ----------
        filename
            Name of the file in which to store the configuration.
        """
        setup = self.to_dict()

        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)
        return

    def var_param(self, var_param: str) -> Param:
        """Returns the value of the given variable parameter name.

        Parameters
        ----------
        var_param
            Name of the variable parameter.

        Returns
        -------
        Value of the specified ``var_param``.
        """
        val = self._var_params.get(var_param)
        if (val is None) and (var_param in self.PARENTS):
            return self.var_param(self.PARENTS[var_param])

        if val is None:
            raise ValueError(f"Variable param {var_param} not in 'Setup.free_params'.")
        return val

    def set_var_param(self, var_param: str, val: Param) -> None:
        """Sets the given value to the given variable parameter.

        Parameters
        ----------
        var_param
            Name of the variable parameter.
        val
            Value to set to ``var_param``.
        """
        if not isinstance(var_param, str):
            raise TypeError(
                f"'var_param' must be a str, but {type(var_param)} was given."
            )
        if not isinstance(val, Param):
            raise TypeError(f"'val' must be {Param}, but {type(val)} was given.")

        self._var_params[var_param] = val
        return

    def set_param(
        self, param: str, param_val: Param, qubits: str | tuple[str, ...] = tuple()
    ) -> None:
        """Sets the given value to the given parameter of the given qubit(s).
        For example, setting the CZ error probability requires
        ``qubits = tuple[str, str]``.

        Parameters
        ----------
        param
            Name of the parameter.
        param_val
            Value to set to ``param``.
        qubits
            Qubit(s) of which to set the parameter.
        """
        if not isinstance(param, str):
            raise TypeError(f"'param' must be a str, but {type(param)} was given.")
        if not isinstance(param_val, Param):
            raise TypeError(
                f"'param_val' must be {Param} but {type(param_val)} was given."
            )
        if isinstance(qubits, str):
            qubits = (qubits,)
        if (not isinstance(qubits, Sequence)) or (
            any(not isinstance(q, str) for q in qubits)
        ):
            raise TypeError(
                f"'qubits' must be a tuple[str], but {type(qubits)} was given."
            )

        qubits = tuple(qubits)
        if not qubits:
            self._global_params[param] = param_val
        else:
            if qubits not in self._qubit_params:
                raise ValueError(
                    f"'{param}' for '{'-'.join(qubits)}' is not a param of this setup."
                )
            self._qubit_params[qubits][param] = param_val
        return

    def param(self, param: str, qubits: str | tuple[str, ...] = tuple()) -> Param:
        """Returns the value of the given parameter for the specified qubit(s).
        For example, getting the CZ error probability requires
        ``qubits = tuple[str, str]``.

        Parameters
        ----------
        param
            Name of the parameter.
        qubits
            Qubit(s) of which to get the parameter.

        Returns
        -------
        val
            Value of the parameter.
        """
        if not isinstance(param, str):
            raise TypeError(f"'param' must be a str, but {type(param)} was given.")
        if isinstance(qubits, str):
            qubits = (qubits,)
        if (not isinstance(qubits, Sequence)) or (
            any(not isinstance(q, str) for q in qubits)
        ):
            raise TypeError(
                f"'qubits' must be a tuple[str], but {type(qubits)} was given."
            )

        qubits = tuple(qubits)
        if qubits in self._qubit_params and param in self._qubit_params[qubits]:
            val = self._qubit_params[qubits][param]
            return self._eval_param_val(val)
        if param in self._global_params:
            val = self._global_params[param]
            return self._eval_param_val(val)

        # if none of the previous works, try loading from 'parent' parameter
        if param in self.PARENTS:
            return self.param(self.PARENTS[param])

        if qubits:
            raise KeyError(
                f"'{param}' for '{'-'.join(qubits)}' is not a param of this setup."
            )
        raise KeyError(f"Global parameter {param} not defined")

    def _eval_param_val(self, val):
        # Parameter values can refer to another parameter (i.e. a variable parameter)
        if not isinstance(val, str):
            return val

        if params := _get_var_params(val):
            for p in params:
                if self._var_params[p] is None:
                    raise ValueError(f"The free param '{p}' has not been specified.")

            # if val = "{parameter}", then no evaluation is needed
            # this is important if the value of 'parameter' is a string because
            # we don't want to do 'eval("value_of_parameter") in this case
            if val == f"{{{params[0]}}}" and isinstance(
                self._var_params[params[0]], str
            ):
                return val.format(**self._var_params)

            val = val.format(**self._var_params)

            # ensure that eval only performs mathematical operations
            val_check = val.replace("True", "").replace("False", "").replace(" ", "")
            if set(val_check) > set("0123456789.*/+-^%~|()=<>?"):
                raise ValueError(
                    "The strings with variable parameters can only be mathematical expressions."
                )
            val = eval(val)

        return val

    def gate_duration(self, name: str) -> float:
        """Returns the duration of the specified gate.

        Parameters
        ----------
        name
            Name of the gate.

        Returns
        -------
        Duration of the gate.
        """
        try:
            return self._gate_durations[name]
        except KeyError:
            raise ValueError(f"No gate duration specified for '{name}'")


def _get_var_params(string: str) -> list[str]:
    params = []
    for s in string.split("{")[1:]:
        if "}" not in s:
            raise ValueError(
                "Only one level of brakets is allowed. Ensure that brakets are matched."
            )

        param = s.split("}")[0]
        if param == "":
            raise ValueError("Params must be non-empty strings.")
        params.append(param)

    return params

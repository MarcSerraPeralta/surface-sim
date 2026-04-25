from __future__ import annotations

from collections.abc import Callable, Collection
from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import yaml

from .random import RandomSetupDict

Param = float | int | bool | str | None


class SetupDict(TypedDict):
    setup: Collection[dict[str, Param]]


ANNOTATIONS = {"tick": "TICK", "qubit_coords": "QUBIT_COORDS"}
SQ_GATES = {
    "idle": "I",
    "x_gate": "X",
    "z_gate": "Z",
    "hadamard": "H",
    "h_gate": "H",
    "s_gate": "S",
    "s_dag_gate": "S_DAG",
    "c_nxyz_gate": "C_NXYZ",
    "c_nzyx_gate": "C_NZYX",
    "c_xnyz_gate": "C_XNYZ",
    "c_xynz_gate": "C_XYNZ",
    "c_xyz_gate": "C_XYZ",
    "c_znyx_gate": "C_ZNYX",
    "c_zynx_gate": "C_ZYNX",
    "c_zyx_gate": "C_ZYX",
    "h_nxy_gate": "H_NXY",
    "h_nxz_gate": "H_NXZ",
    "h_nyz_gate": "H_NYZ",
    "h_xy_gate": "H_XY",
    "h_xz_gate": "H",  # stim changes the name
    "h_yz_gate": "H_YZ",
    "sqrt_x_gate": "SQRT_X",
    "sqrt_x_dag_gate": "SQRT_X_DAG",
    "sqrt_y_gate": "SQRT_Y",
    "sqrt_y_dag_gate": "SQRT_Y_DAG",
    "sqrt_z_gate": "S",  # stim changes the name
    "sqrt_z_dag_gate": "S_DAG",  # stim changes the name
}
TQ_GATES = {
    "cnot": "CX",  # stim changes the name
    "cx": "CX",
    "cxswap": "CXSWAP",
    "cy": "CY",
    "cphase": "CZ",
    "cz": "CZ",
    "czswap": "CZSWAP",
    "idleidle": "II",
    "iswap": "ISWAP",
    "iswap_dag": "ISWAP_DAG",
    "sqrt_xx": "SQRT_XX",
    "sqrt_xx_dag": "SQRT_XX_DAG",
    "sqrt_yy": "SQRT_YY",
    "sqrt_yy_dag": "SQRT_YY_DAG",
    "sqrt_zz": "SQRT_ZZ",
    "sqrt_zz_dag": "SQRT_ZZ_DAG",
    "swap": "SWAP",
    "swapcx": "SWAPCX",
    "swapcz": "CZSWAP",  # stim changes the name
    "xcx": "XCX",
    "xcy": "XCY",
    "xcz": "XCZ",
    "ycx": "YCX",
    "ycy": "YCY",
    "ycz": "YCZ",
    "zcx": "CX",  # stim changes the name
    "zcy": "CY",  # stim changes the name
    "zcz": "CZ",  # stim changes the name
}
LONG_RANGE_TQ_GATES = {f"long_range_{k}": v for k, v in TQ_GATES.items()}
SQ_MEASUREMENTS = {
    "measure": "M",
    "measure_x": "MX",
    "measure_y": "MY",
    "measure_z": "M",  # stim changes the name
}
SQ_RESETS = {
    "reset": "R",
    "reset_x": "RX",
    "reset_y": "RY",
    "reset_z": "R",  # stim changes the name
}

SQ_PARENTS = (
    {f"{n}_error_prob": "sq_error_prob" for n in SQ_GATES}
    | {f"{n}_error_prob": "reset_error_prob" for n in SQ_RESETS}
    | {f"{n}_error_prob": "meas_error_prob" for n in SQ_MEASUREMENTS}
)
TQ_PARENTS = {f"{n}_error_prob": "tq_error_prob" for n in TQ_GATES} | {
    f"{n}_error_prob": "long_range_tq_error_prob" for n in LONG_RANGE_TQ_GATES
}
PARENTS = SQ_PARENTS | TQ_PARENTS

SQ_PARAMS = set(SQ_PARENTS) | set(SQ_PARENTS.values())
TQ_PARAMS = set(TQ_PARENTS) | set(TQ_PARENTS.values())


class Setup:
    PARENTS: dict[str, str] = PARENTS.copy()

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
        self._mode: str = "standard"
        self._qubit_params: dict[str | tuple[str, ...], dict[str, Param]] = dict()
        self._global_params: dict[str, Param] = dict()
        self._var_params: dict[str, Param] = dict()
        self.uniform: bool = False

        _setup: SetupDict = deepcopy(setup)
        self.name: str | None = _setup.pop("name", None)
        self.description: str | None = _setup.pop("description", None)
        self._gate_durations: dict[str, float | int] = _setup.pop("gate_durations", {})
        self._load_setup(_setup)
        if self._qubit_params == {}:
            self.uniform = True

        # random samplers
        self._free_param_samplers: dict[str, Callable[[], Param]] = {}
        self._standard_qubit_params: dict[str | tuple[str, ...], dict[str, Param]] = (
            dict()
        )

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
        """Returns the names of the unset variable parameters."""
        return [param for param, val in self._var_params.items() if val is None]

    @property
    def var_params(self) -> list[str]:
        """Returns the names of all variable parameters."""
        return list(self._var_params)

    @property
    def global_params(self) -> list[str]:
        """Returns the names of the global parameters."""
        return list(self._global_params)

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
            setup: SetupDict = yaml.safe_load(file)
            return cls(setup)

    def to_dict(self) -> SetupDict:
        """Returns a dictionary that can be used to initialize ``Setup``."""
        setup = dict()

        setup["name"] = self.name
        setup["description"] = self.description
        setup["gate_durations"] = self._gate_durations

        qubit_params: list[dict[str, Param]] = []
        if self._global_params:
            qubit_params.append(self._global_params)

        for qubits, params in self._qubit_params.items():
            params_copy = deepcopy(params)
            num_qubits = len(qubits)
            if num_qubits == 1:
                params_copy["qubit"] = qubits[0]
            elif num_qubits == 2:
                params_copy["qubits"] = tuple(qubits)
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

    def convert_to_random(self, **free_param_samplers: Callable[[], Param]) -> None:
        """Converts this setup into a random setup, in which the free parameters
        are randomly sampled for each qubit and qubit pair.

        The free parameters (``Setup.free_params``) are sampled on the fly the
        first time they are requested and then stored. If a qubit or qubit pair has
        some parameters already set, their value is not modified, that is: no free
        parameter is sampled for this qubit or qubit pair.
        Because the free parameters are sampled on the fly, the setup must be
        stored **after** generating the wanted circuit. Otherwise, the parameters
        are not yet sampled nor stored in this setup.

        The qubit-specific parameters that are generated from the randomly sampled
        values of the free parameter correspond to the global parameters of this
        setup (``Setup.global_params``).

        Parameters
        ----------
        **free_param_samplers
            Samplers for the free parameters.
        """
        if self._mode != "standard":
            raise ValueError(
                f"Setup must be in 'standard' mode, but it is in '{self._mode}'."
            )
        if set(free_param_samplers) < set(self.free_params):
            raise ValueError(
                f"All free parameters ({', '.join(self.free_params)}) must be specified."
            )

        self._mode = "random"
        self._free_param_samplers = dict(free_param_samplers)
        self._standard_qubit_params = dict(self._qubit_params)
        self._qubit_params = RandomSetupDict(self._standard_qubit_params)
        self.uniform = False

        global_params_copy = dict(self._global_params)
        var_params_copy = dict(self._var_params)
        free_param_samplers_copy = dict(free_param_samplers)

        def sq_noise_sampler() -> dict[str, Param]:
            free_params = {p: s() for p, s in free_param_samplers_copy.items()}
            var_params = dict(var_params_copy)
            var_params.update(free_params)

            qubit_params: dict[str, Param] = {}
            for name, value in global_params_copy.items():
                if name not in SQ_PARAMS:
                    continue

                qubit_params[name] = _eval_param_val(value, var_params)
            return qubit_params

        def tq_noise_sampler() -> dict[str, Param]:
            free_params = {p: s() for p, s in free_param_samplers_copy.items()}
            var_params = dict(var_params_copy)
            var_params.update(free_params)

            qubit_params: dict[str, Param] = {}
            for name, value in global_params_copy.items():
                if name not in TQ_PARAMS:
                    continue

                qubit_params[name] = _eval_param_val(value, var_params)
            return qubit_params

        self._qubit_params.sq_noise_sampler = sq_noise_sampler
        self._qubit_params.tq_noise_sampler = tq_noise_sampler

        return

    def new_randomization(self) -> None:
        """Restores the original setup without the sampled parameters."""
        if self._mode != "random":
            raise ValueError(
                f"Setup must be in 'random' mode, but it is in '{self._mode}'."
            )

        self._qubit_params = RandomSetupDict(self._standard_qubit_params)
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
        if self._mode == "random":
            raise ValueError("Parameters cannot be changed in 'random' mode.")
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
        if self._mode == "random":
            raise ValueError("Parameters cannot be changed in 'random' mode.")
        if not isinstance(param, str):
            raise TypeError(f"'param' must be a str, but {type(param)} was given.")
        if not isinstance(param_val, Param):
            raise TypeError(
                f"'param_val' must be {Param} but {type(param_val)} was given."
            )
        if isinstance(qubits, str):
            qubits = (qubits,)
        if (not isinstance(qubits, Collection)) or (
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

    def param(self, param: str, qubits: str | Collection[str] = tuple()) -> Param:
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
        if (not isinstance(qubits, Collection)) or (
            any(not isinstance(q, str) for q in qubits)
        ):
            raise TypeError(
                f"'qubits' must be a tuple[str], but {type(qubits)} was given."
            )

        qubits = tuple(qubits)

        if self._mode == "standard":
            if qubits in self._qubit_params and param in self._qubit_params[qubits]:
                val = self._qubit_params[qubits][param]
                return _eval_param_val(val, self._var_params)
            if param in self._global_params:
                val = self._global_params[param]
                return _eval_param_val(val, self._var_params)
        elif self._mode == "random":
            if len(qubits) == 0:
                raise ValueError("In 'random' mode, 'qubits' must be specified.")
            params = self._qubit_params[qubits]
            if param in params:
                # parameter's value has already been evaluated
                return params[param]

        # if none of the previous works, try loading from 'parent' parameter
        if param in self.PARENTS:
            return self.param(self.PARENTS[param], qubits=qubits)

        if qubits:
            raise KeyError(
                f"'{param}' for '{'-'.join(qubits)}' is not a param of this setup."
            )
        raise KeyError(f"Global parameter {param} not defined")

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
    params: list[str] = []
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


def _eval_param_val(val: Param, var_params: dict[str, Param]) -> Param:
    # Parameter values can refer to another parameter (i.e. a variable parameter)
    if not isinstance(val, str):
        return val

    if params := _get_var_params(val):
        for p in params:
            if var_params[p] is None:
                raise ValueError(f"The free param '{p}' has not been specified.")

        # if val = "{parameter}", then no evaluation is needed
        # this is important if the value of 'parameter' is a string because
        # we don't want to do 'eval("value_of_parameter") in this case
        if val == f"{{{params[0]}}}" and isinstance(var_params[params[0]], str):
            return val.format(**var_params)

        val = val.format(**var_params)

        # ensure that eval only performs mathematical operations
        val_check = val.replace("True", "").replace("False", "").replace(" ", "")
        if set(val_check) > set("0123456789.*/+-^%~|()=<>?"):
            raise ValueError(
                "The strings with variable parameters can only be mathematical expressions."
            )
        val = eval(val)

    return val

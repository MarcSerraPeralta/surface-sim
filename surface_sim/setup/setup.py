from typing import Any, Dict, Type, TypeVar, Union, List
from copy import deepcopy
from pathlib import Path

import yaml

T = TypeVar("T", bound="Setup")


class Setup:
    def __init__(self, setup: Dict[str, Any]) -> None:
        self._qubit_params = dict()
        self._global_params = dict()
        self._var_params = dict()

        _setup = deepcopy(setup)
        self.name = _setup.pop("name", None)
        self.description = _setup.pop("description", None)
        self._gate_durations = _setup.pop("gate_durations", {})
        self._load_setup(_setup)

    def _load_setup(self, setup: Dict[str, Any]) -> None:
        params = setup.get("setup")
        if not params:
            raise ValueError("setup not found or contains no information")

        for params_dict in params:
            if "qubit" in params_dict:
                qubit = str(params_dict.pop("qubit"))
                qubits = (qubit,)
            elif "qubits" in params_dict:
                qubits = tuple(params_dict.pop("qubits"))
            else:
                qubits = None

            if qubits:
                if qubits in self._qubit_params.keys():
                    raise ValueError("Parameters defined repeatedly in the setup.")
                self._qubit_params[qubits] = params_dict
            else:
                self._global_params.update(params_dict)

            for param, val in params_dict.items():
                if isinstance(val, str) and param not in self._var_params:
                    self._var_params[val] = None

    @property
    def free_params(self) -> List[str]:
        return [param for param, val in self._var_params.items() if val is None]

    @classmethod
    def from_yaml(cls: Type[T], filename: Union[str, Path]) -> T:
        """
        from_yaml Create new surface_sim.setup.Setup instance from YAML configuarion file.

        Parameters
        ----------
        filename : str
            The YAML file name.

        Returns
        -------
        T
            The initialised surface_sim.setup.Setup object based on the yaml.
        """
        with open(filename, "r") as file:
            setup = yaml.safe_load(file)
            return cls(setup)

    def to_dict(self) -> Dict[str, Any]:
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

    def to_yaml(self, filename: Union[str, Path]) -> None:
        setup = self.to_dict()

        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)

    def var_param(self, var_param: str) -> float:
        try:
            return self._var_params[var_param]
        except KeyError:
            raise ValueError(f"Variable param {var_param} not in setup.free_params.")

    def set_var_param(self, var_param: str, val: float) -> None:
        try:
            self._var_params[var_param] = val
        except KeyError:
            raise ValueError(f"Variable param {var_param} not in setup.")

    def set_param(self, param: str, param_val: float, *qubits: str) -> None:
        if not qubits:
            self._global_params[param] = param_val
        else:
            self._qubit_params[qubits][param] = param_val

    def param(self, param: str, *qubits: str) -> float:
        if qubits in self._qubit_params and param in self._qubit_params[qubits]:
            val = self._qubit_params[qubits][param]
            return self._eval_param_val(val)
        if param in self._global_params:
            val = self._global_params[param]
            return self._eval_param_val(val)

        if qubits:
            qubit_str = ", ".join(qubits)
            raise KeyError(f"Parameter {param} not defined for qubit(s) {qubit_str}")
        raise KeyError(f"Global parameter {param} not defined")

    def _eval_param_val(self, val):
        if val in self._var_params:
            param = deepcopy(val)
            val = self._var_params[param]
            if val is None:
                raise ValueError(f"Variable parameter {param} is not specified.")

        return val

    def gate_duration(self, name: str) -> float:
        try:
            return self._gate_durations[name]
        except KeyError:
            raise ValueError(f"No gate duration specified for '{name}'")

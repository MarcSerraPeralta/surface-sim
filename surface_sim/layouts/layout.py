from __future__ import annotations

from copy import deepcopy
from os import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union

import networkx as nx
import numpy as np
import yaml
from xarray import DataArray

IntDirections = List[str]
IntOrder = Union[IntDirections, Dict[str, IntDirections]]


class Layout:
    """Layout class for a QEC code.

    Initialization and storage
    --------------------------
    - ``__init__``
    - ``__copy__``
    - ``to_dict``
    - ``from_yaml``
    - ``to_yaml``

    Get information
    ---------------
    - ``param``
    - ``get_inds``
    - ``get_qubits``
    - ``get_neighbors``
    - ``get_coords``

    Set information
    ---------------
    - ``set_param``

    Matrix generation
    -----------------
    - ``adjacency_matrix``
    - ``expansion_matrix``
    - ``projection_matrix``
    - ``stab_gen_matrix``

    """

    def __init__(self, setup: Dict[str, Any]) -> None:
        """Initiailizes the layout.

        Parameters
        ----------
        setup
            The layout setup, provided as a dict.

            The setup dictionary is expected to have a 'layout' item, containing
            a list of dictionaries. Each such dictionary (``dict[str, Any]``) must define the
            qubit label (``str``) corresponding the ``'qubit'`` item. In addition, each dictionary
            must also have a ``'neighbors'`` item that defines a dictonary (``dict[str, str]``)
            of ordinal directions and neighbouring qubit labels. Apart from these two items,
            each dictionary can hold any other metadata or parameter relevant to these qubits.

            In addition to the layout list, the setup dictionary can also optionally
            define the name of the layout (``str``), a description (``str``) of the layout as well
            as the interaction order of the different types of check, if the layout is used
            for a QEC code.

        Raises
        ------
        ValueError
            If the type of the setup provided is not a dictionary.
        """
        if not isinstance(setup, dict):
            raise ValueError(f"'setup' must be a dict, instead got {type(setup)}.")

        self.name = setup.get("name")
        self.code = setup.get("code", "")
        self.distance = setup.get("distance", -1)
        self.distance_z = setup.get("distance_z", -1)
        self.distance_x = setup.get("distance_x", -1)
        self.log_z = setup.get("log_z", [])
        self.log_x = setup.get("log_x", [])
        self.description = setup.get("description")
        self.interaction_order = setup.get("interaction_order", {})

        self.graph = nx.DiGraph()
        self._load_layout(setup)

        qubits = list(self.graph.nodes)
        self._qubit_inds = dict(zip(qubits, range(len(qubits))))

    def __copy__(self) -> Layout:
        """Copies the Layout."""
        return Layout(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Return a setup dictonary for the layout.

        Returns
        -------
        setup
            The dictionary of the setup.
            A copyt of this ``Layout`` can be initalized using ``Layout(setup)``.
        """
        setup = dict()

        setup["name"] = self.name
        setup["code"] = self.code
        setup["distance"] = self.distance
        setup["distance_z"] = self.distance_z
        setup["distance_x"] = self.distance_x
        setup["log_z"] = self.log_z
        setup["log_x"] = self.log_x
        setup["description"] = self.description
        setup["interaction_order"] = self.interaction_order

        layout = []
        for node, attrs in self.graph.nodes(data=True):
            node_dict = deepcopy(attrs)
            node_dict["qubit"] = node

            nbr_dict = dict()
            adj_view = self.graph.adj[node]

            for nbr_node, edge_attrs in adj_view.items():
                edge_dir = edge_attrs["direction"]
                nbr_dict[edge_dir] = nbr_node

            for ver_dir in ("north", "south"):
                for hor_dir in ("east", "west"):
                    edge_dir = f"{ver_dir}_{hor_dir}"
                    if edge_dir not in nbr_dict:
                        nbr_dict[edge_dir] = None

            node_dict["neighbors"] = nbr_dict

            layout.append(node_dict)
        setup["layout"] = layout
        return setup

    def get_inds(self, qubits: Iterable[str]) -> List[int]:
        """Returns the indices of the qubits.

        Parameters
        ----------
        qubits
            List of qubits.

        Returns
        -------
        The list of qubit indices.
        """
        return [self._qubit_inds[qubit] for qubit in qubits]

    def get_qubits(self, **conds: Any) -> List[str]:
        """Return the qubit labels that meet a set of conditions.

        Parameters
        ----------
        **conds
            Dictionary of the conditions.

        Returns
        -------
        nodes
            The list of qubit labels that meet all conditions.

        Notes
        -----
        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (``Any``)
        that each parameter label (``str``) needs to take.
        """
        if conds:
            node_view = self.graph.nodes(data=True)
            nodes = [node for node, attrs in node_view if valid_attrs(attrs, **conds)]
            return nodes

        nodes = list(self.graph.nodes)
        return nodes

    def get_neighbors(
        self,
        qubits: List[str],
        direction: Optional[str] = None,
        as_pairs: bool = False,
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Returns the list of qubit labels, neighboring specific qubits
        that meet a set of conditions.

        Parameters
        ----------
        qubits
            The qubit labels, whose neighbors are being considered.

        direction
            The direction along which to consider the neigbors along.

        Returns
        -------
        end_notes
            The list of qubit label, neighboring qubit, that meet the conditions.

        Notes
        -----
        The order that the qubits appear in is defined during the initialization
        of the layout and remains fixed.

        The conditions conds are the keyward arguments that specify the value (``Any``)
        that each parameter label (``str``) needs to take.
        """
        edge_view = self.graph.out_edges(qubits, data=True)

        start_nodes = []
        end_nodes = []
        for start_node, end_node, attrs in edge_view:
            if direction is None or attrs["direction"] == direction:
                start_nodes.append(start_node)
                end_nodes.append(end_node)

        if as_pairs:
            return list(zip(start_nodes, end_nodes))
        return end_nodes

    def get_coords(self, qubits: List[str]) -> List[List[float | int]]:
        """Returns the coordinates of the given qubits.

        Parameters
        ----------
        qubits
            List of qubits.

        Returns
        -------
        Coordinates of the given qubits.
        """
        all_coords = nx.get_node_attributes(self.graph, "coords")

        if set(qubits) > set(all_coords):
            raise ValueError("Some of the given qubits do not have coordinates.")

        return [all_coords[q] for q in qubits]

    def adjacency_matrix(self) -> DataArray:
        """Returns the adjaceny matrix corresponding to the layout.

        The layout is encoded as a directed graph, such that there are two edges
        in opposite directions between each pair of neighboring qubits.

        Returns
        -------
        ajd_matrix
            The adjacency matrix.
        """
        qubits = self.get_qubits()
        adj_matrix = nx.adjacency_matrix(self.graph)

        data_arr = DataArray(
            data=adj_matrix.toarray(),
            dims=["from_qubit", "to_qubit"],
            coords=dict(
                from_qubit=qubits,
                to_qubit=qubits,
            ),
        )
        return data_arr

    def expansion_matrix(self) -> DataArray:
        """Returns the expansion matrix corresponding to the layout.
        The matrix can expand a vector of measurements/defects to a 2D array corresponding to layout of the ancilla qubits. Used for convolutional neural networks.

        Returns
        -------
        DataArray
            The expansion matrix.
        """
        node_view = self.graph.nodes(data=True)

        anc_qubits = [node for node, data in node_view if data["role"] == "anc"]
        coords = [node_view[anc]["coords"] for anc in anc_qubits]

        rows, cols = zip(*coords)

        row_inds, num_rows = index_coords(rows, reverse=True)
        col_inds, num_cols = index_coords(cols)

        num_anc = len(anc_qubits)
        anc_inds = range(num_anc)

        tensor = np.zeros((num_anc, num_rows, num_cols), dtype=bool)
        tensor[anc_inds, row_inds, col_inds] = True
        expanded_tensor = np.expand_dims(tensor, axis=1)

        expansion_tensor = DataArray(
            expanded_tensor,
            dims=["anc_qubit", "channel", "row", "col"],
            coords=dict(
                anc_qubit=anc_qubits,
            ),
        )
        return expansion_tensor

    def projection_matrix(self, stab_type: str) -> DataArray:
        """Returns the projection matrix, mapping
        data qubits (defined by a parameter 'role' equal to 'data')
        to ancilla qubits (defined by a parameter 'role' equal to 'anc')
        measuing a given stabilizerr type (defined by a parameter
        'stab_type' equal to stab_type).

        This matrix can be used to project a final set of data-qubit
        measurements to a set of syndromes.

        Parameters
        ----------
        stab_type : str
            The type of the stabilizers that the data qubit measurement
            is being projected to.

        Returns
        -------
        DataArray
            The projection matrix.
        """
        adj_mat = self.adjacency_matrix()

        anc_qubits = self.get_qubits(role="anc", stab_type=stab_type)
        data_qubits = self.get_qubits(role="data")

        proj_mat = adj_mat.sel(from_qubit=data_qubits, to_qubit=anc_qubits)
        return proj_mat.rename(from_qubit="data_qubit", to_qubit="anc_qubit")

    def stab_gen_matrix(self, log_gate: str) -> DataArray:
        """Returns the unitary matrix that specified how the stabilizer
        generators are transformed in the specified logical gate.

        See module ``surface_sim.log_gates`` to see how to prepare
        the layout to run logical gates.

        Parameters
        ---------
        log_gate
            Logical gate.

        Returns
        -------
        unitary_mat
            Unitary matrix specifying the transformation of the stabilizer
            generators. Its coordinates are ``stab_gen`` and ``new_stab_gen``.
        """
        anc_qubits = self.get_qubits(role="anc")
        new_stab_gens = []
        for anc_qubit in anc_qubits:
            log_gate_attrs = self.param(log_gate, anc_qubit)
            if log_gate_attrs is None:
                raise ValueError(
                    f"New stabilizer generators for {log_gate} "
                    f"are not specified for qubit {anc_qubit}."
                    "They should be setted with 'surface_sim.log_gates'."
                )
            new_stab_gen = log_gate_attrs["new_stab_gen"]
            new_stab_gen_inds = [anc_qubits.index(q) for q in new_stab_gen]

            new_stab_gen_array = np.zeros(len(anc_qubits))
            new_stab_gen_array[new_stab_gen_inds] = 1
            new_stab_gens.append(new_stab_gen_array)

        unitary_mat = DataArray(
            data=new_stab_gens,
            coords=dict(new_stab_gen=anc_qubits, stab_gen=anc_qubits),
        )
        # galois requires that the arrays are integers, not floats.
        unitary_mat = unitary_mat.astype(int)

        return unitary_mat

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]) -> "Layout":
        """Loads the layout class from a YAML file.

        The file must define the setup dictionary that initializes
        the layout.

        Parameters
        ----------
        filename : Union[str, Path]
            The pathfile name of the YAML setup file.

        Returns
        -------
        Layout
            The initialized layout object.

        Raises
        ------
        ValueError
            If the specified file does not exist.
        ValueError
            If the specified file is not a string.
        """
        if not path.exists(filename):
            raise ValueError("Given path doesn't exist")

        with open(filename, "r") as file:
            layout_setup = yaml.safe_load(file)
            return cls(layout_setup)

    def to_yaml(self, filename: Union[str, Path]) -> None:
        """Saves the layout as a YAML file.

        Parameters
        ----------
        filename : Union[str, Path]
            The pathfile name of the YAML setup file.

        """
        setup = self.to_dict()
        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)

    def param(self, param: str, qubit: str) -> Any:
        """Returns the parameter value of a given qubit

        Parameters
        ----------
        param
            The label of the qubit parameter.
        qubit
            The label of the qubit that is being queried.

        Returns
        -------
        Any
            The value of the parameter if specified for the given qubit,
            else ``None``.
        """
        if param not in self.graph.nodes[qubit]:
            return None
        else:
            return self.graph.nodes[qubit][param]

    def set_param(self, param: str, qubit: str, value: Any) -> None:
        """Sets the value of a given qubit parameter

        Parameters
        ----------
        param : str
            The label of the qubit parameter.
        qubit : str
            The label of the qubit that is being queried.
        value : Any
            The new value of the qubit parameter.
        """
        self.graph.nodes[qubit][param] = value

    def _load_layout(self, setup: Dict[str, Any]) -> None:
        """Internal function that loads the directed graph from the
        setup dictionary that is provided during initialization.

        Parameters
        ----------
        setup : Dict[str, Any]
            The setup dictionary that must specify the 'layout' list
            of dictionaries, containing the qubit informaiton.

        Raises
        ------
        ValueError
            If there are unlabeled qubits in the any of the layout dictionaries.
        ValueError
            If any qubit label is repeated in the layout list.
        """
        layout = deepcopy(setup.get("layout"))
        if layout is None:
            raise ValueError("'setup' does not contain a 'layout' key.")

        for qubit_info in layout:
            qubit = qubit_info.pop("qubit", None)
            if qubit is None:
                raise ValueError("Each qubit in the layout must be labeled.")

            if qubit in self.graph:
                raise ValueError("Qubit label repeated, ensure labels are unique.")

            self.graph.add_node(qubit, **qubit_info)

        for node, attrs in self.graph.nodes(data=True):
            nbr_dict = attrs.get("neighbors", None)
            for edge_dir, nbr_qubit in nbr_dict.items():
                if nbr_qubit is not None:
                    self.graph.add_edge(node, nbr_qubit, direction=edge_dir)


def valid_attrs(attrs: Dict[str, Any], **conditions: Any) -> bool:
    """Checks if the items in attrs match each condition in conditions.
    Both attrs and conditions are dictionaries mapping parameter labels (str)
    to values (Any).

    Parameters
    ----------
    attrs : Dict[str, Any]
        The attribute dictionary.

    Returns
    -------
    bool
        Whether the attributes meet a set of conditions.
    """
    for key, val in conditions.items():
        attr_val = attrs[key]
        if attr_val is None or attr_val != val:
            return False
    return True


def index_coords(coords: List[int], reverse: bool = False) -> Tuple[List[int], int]:
    """Indexes a list of coordinates.

    Parameters
    ----------
    coords : List[int]
        The list of coordinates.
    reverse : bool, optional
        Whether to return the values in reverse, by default False

    Returns
    -------
    Tuple[List[int], int]
        The list of indexed coordinates and the number of unique coordinates.
    """
    unique_vals = set(coords)
    num_unique_vals = len(unique_vals)

    if reverse:
        unique_inds = reversed(range(num_unique_vals))
    else:
        unique_inds = range(num_unique_vals)

    mapping = dict(zip(unique_vals, unique_inds))

    indicies = [mapping[coord] for coord in coords]
    return indicies, num_unique_vals

import xarray as xr

from surface_sim import Layout


LAYOUT_DICT = {
    "name": "Rotated dx-1 dz-2 surface code layout.",
    "code": "rotated_surface_code",
    "logical_qubits": {
        "L0": {
            "ind": 0,
            "log_z": ["D1", "D2"],
            "log_x": ["D1"],
        }
    },
    "observables": {"O0": ["X1"]},
    "distance_z": 2,
    "distance_x": 1,
    "description": None,
    "interaction_order": {
        "x_type": ["north_east", "north_west", "south_east", "south_west"],
        "z_type": ["north_east", "south_east", "north_west", "south_west"],
    },
    "layout": [
        {
            "role": "data",
            "coords": [1, 1],
            "freq_group": "low",
            "stab_type": None,
            "ind": 0,
            "neighbors": {
                "south_east": "X1",
            },
            "qubit": "D1",
        },
        {
            "role": "data",
            "coords": [1, 3],
            "freq_group": "low",
            "stab_type": None,
            "ind": 1,
            "neighbors": {
                "south_west": "X1",
            },
            "qubit": "D2",
        },
        {
            "role": "anc",
            "coords": [0, 2],
            "freq_group": "mid",
            "stab_type": "x_type",
            "ind": 2,
            "neighbors": {
                "north_east": "D2",
                "north_west": "D1",
            },
            "qubit": "X1",
        },
    ],
}


def test_layout():
    layout = Layout(LAYOUT_DICT)

    assert isinstance(layout, Layout)

    return


def test_layout_input_output(tmp_path):
    layout = Layout(LAYOUT_DICT)

    assert layout.to_dict() == LAYOUT_DICT

    loaded_layout = Layout.from_dict(layout.to_dict())
    assert layout.to_dict() == loaded_layout.to_dict()

    file_name = tmp_path / "test.yaml"
    layout.to_yaml(file_name)
    loaded_layout = Layout.from_yaml(file_name)
    assert layout.to_dict() == loaded_layout.to_dict()

    return


def test_layout_get_information():
    layout = Layout(LAYOUT_DICT)

    assert layout.param("fdsakljsfdj", "D1") is None
    assert layout.param("role", "D1") == "data"

    assert layout.get_inds(["X1", "D2"]) == (2, 1)
    assert layout.qubit_inds == {"D1": 0, "D2": 1, "X1": 2}
    assert layout.num_qubits == 3

    assert layout.get_max_ind() == 2
    assert layout.get_min_ind() == 0

    assert layout.get_labels_from_inds([2]) == ("X1",)

    assert layout.data_qubits == ("D1", "D2")
    assert layout.num_data_qubits == 2

    assert layout.anc_qubits == ("X1",)
    assert layout.num_anc_qubits == 1
    assert layout.get_qubits(role="anc", stab_type="z_type") == tuple()

    assert layout.logical_qubits == ("L0",)
    assert layout.num_logical_qubits == 1

    assert layout.observables == ("O0",)
    assert layout.num_observables == 1
    assert layout.observable_definition("O0") == ("X1",)

    assert set(layout.get_neighbors(["D1", "X1"])) == set(["X1", "D1", "D2"])
    assert set(layout.get_neighbors(["D1"], as_pairs=True)) == set([("D1", "X1")])
    assert set(layout.get_neighbors(["D1"], direction="south_east")) == set(["X1"])

    assert set(layout.get_support(["D1"])["D1"]) == set(["X1"])

    assert layout.get_coords(["X1", "D2"]) == ((0, 2), (1, 3))

    assert layout.qubit_coords == {"D1": (1, 1), "D2": (1, 3), "X1": (0, 2)}

    assert layout.anc_coords == {"X1": (0, 2)}

    return


def test_layout_set_information():
    layout = Layout(LAYOUT_DICT)

    layout.set_param("role", "D1", "abcdef")
    assert layout.param("role", "D1") == "abcdef"

    layout.set_logical_param("ind", "L0", "abcdef")
    assert layout.logical_param("ind", "L0") == "abcdef"

    return


def test_layout_matrices():
    layout = Layout(LAYOUT_DICT)

    expected_matrix = xr.DataArray(
        data=[[0, 0, 1], [0, 0, 1], [1, 1, 0]],
        coords=dict(from_qubit=["D1", "D2", "X1"], to_qubit=["D1", "D2", "X1"]),
    )
    assert (layout.adjacency_matrix() == expected_matrix).all()

    expected_matrix = xr.DataArray(
        data=[[1], [1]],
        coords=dict(data_qubit=["D1", "D2"], anc_qubit=["X1"]),
    )
    assert (layout.projection_matrix(stab_type="x_type") == expected_matrix).all()

    expected_matrix = xr.DataArray(
        data=[[[[True]]]],
        coords=dict(anc_qubit=["X1"]),
        dims=["anc_qubit", "channel", "row", "col"],
    )
    assert (layout.expansion_matrix() == expected_matrix).all()

    return

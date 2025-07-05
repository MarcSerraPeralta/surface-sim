from collections.abc import Iterable
from copy import deepcopy
import re

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .layout import Layout


Coordinates = tuple[float, float]
CoordRange = tuple[float, float]

# Regex to filter qubit labels for TeX text.
RE_FILTER = re.compile("([a-zA-Z]+)([0-9]+)")

# Order in which to draw the elements.
# We want the text to be the last drawn object so that is it on top and thus readable.
# Wedge is used to draw the logical support.
ZORDERS = dict(circle=3, wedge=2, patch=0, line=1, text=4)

# Define the default colors
COLORS = {
    "red": "#e41a1cff",
    "red_light": "#f07f80ff",
    "green": "#4daf4aff",
    "green_light": "#9dd49bff",
    "blue": "#377eb8ff",
    "blue_light": "#90bbdfff",
    "orange": "#ff9933ff",
    "orange_light": "#ffb770ff",
    "purple": "#984ea3ff",
    "purple_light": "#ca9ed1ff",
    "yellow": "#f2c829ff",
    "yellow_light": "#f7dc78ff",
}


def clockwise_sort(coordinates: Iterable[Coordinates]) -> list[Coordinates]:
    """Sorts a sequence of coordinates in clockwise order.

    This function is used to correcly draw a ``matplotlib.patches.Polygon``.

    Parameters
    ----------
    coordinates
        The coordinates to sort.

    Returns
    -------
    sorted_coords
        The sorted coordinates.
    """
    coords = list(coordinates)
    x_coords, y_coords = zip(*coords)

    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)

    x_vectors = [x - x_center for x in x_coords]
    y_vectors = [y - y_center for y in y_coords]

    angles = np.arctan2(x_vectors, y_vectors)
    inds = np.argsort(angles)
    sorted_coords = [coords[ind] for ind in inds]
    return sorted_coords


def get_label(qubit: str, coords: Coordinates, **kwargs) -> Text:
    """Draws the label of a qubit.

    Parameters
    ----------
    qubit
        The qubit label.
    coords
        The coordinates of the qubit.
    **kargs
        Extra arguments for ``matplotlib.text.Text``.
    """
    text = deepcopy(qubit)

    # check if the qubit can be plotted using TeX
    match = RE_FILTER.match(qubit)
    if match is not None:
        name, ind = match.groups()
        text = f"${name}_\\mathrm{{{ind}}}$"

    x, y = coords
    zorder = ZORDERS["text"]
    label = Text(x, y, text, zorder=zorder, **kwargs)
    return label


def get_circle(center: Coordinates, radius: float, **kwargs) -> Circle:
    """Draws a ``matplotlib.patches.Circle`` with the given specifications.

    Parameters
    ----------
    coords
        The coordinates of the centre of the circle.
    radius
        The radius of the circle.
    **kargs
        Extra arguments for ``matplotlib.patches.Circle``.

    Returns
    -------
    Circle
        The circle with the given specifications.
    """
    zorder = ZORDERS["circle"]
    circle = Circle(center, radius=radius, zorder=zorder, **kwargs)
    return circle


def get_wedge(
    center: Coordinates, r: float, theta1: float, theta2: float, **kwargs
) -> Wedge:
    """Draws a ``matplotlib.patches.Wedge`` with the given specifications.

    Parameters
    ----------
    coords
        The coordinates of the centre of the circle.
    r
        The radius of the circle.
    theta1
        The angle to start drawing the wedge.
    theta2
        The angle to finish drawing the wedge.
    **kargs
        Extra arguments for ``matplotlib.patches.Wedge``.

    Returns
    -------
    Wedge
        The circle with the given specifications.
    """
    zorder = ZORDERS["wedge"]
    circle = Wedge(center, r=r, theta1=theta1, theta2=theta2, zorder=zorder, **kwargs)
    return circle


def get_patch(patch_coords: Iterable[Coordinates], **kwargs) -> Polygon:
    """Draws a ``matplotlib.patches.Polygon`` with the given specifications.

    Parameters
    ----------
    patch_coords
        The coordinates of the patch.
    **kargs
        Extra arguments for ``matplotlib.patches.Polygon``.
    """
    zorder = ZORDERS["patch"]
    patch = Polygon(patch_coords, closed=True, zorder=zorder, **kwargs)
    return patch


def get_line(coordinates: Iterable[Coordinates], **kwargs) -> Line2D:
    """Draws a connection between two qubits.

    Parameters
    ----------
    qubit_coords
        The coordinates of the qubits.
    **kargs
        Extra arguments for ``matplotlib.lines.Line2D``.

    Returns
    -------
    line
        Line between the two qubits.
    """
    x_coords, y_coords = zip(*coordinates)
    zorder = ZORDERS["line"]
    line = Line2D(x_coords, y_coords, zorder=zorder, **kwargs)
    return line


def qubit_labels(layout: Layout, label_fontsize: float | int = 11) -> Iterable[Text]:
    """Draws the qubit labels from a layout.

    Parameters
    ----------
    layout
        The layout to draw the connections of.
    label_fontsize
        Default value of the font size for the labels.
    """
    default_params = dict(
        color="black",
        verticalalignment="center",
        horizontalalignment="center",
        fontsize=label_fontsize,
    )
    for qubit in layout.qubits:
        coords = layout.param("coords", qubit)
        if len(coords) != 2:
            raise ValueError(
                "Coordinates must be 2D to be plotted, "
                f"but {len(coords)}D were given for qubit {qubit}"
            )
        metaparams = layout.param("metaparams", qubit)
        text_params = deepcopy(default_params)

        if isinstance(metaparams, dict):
            custom_params = metaparams.get("text", {})
            text_params.update(custom_params)

        yield get_label(qubit, coords, **text_params)


def qubit_connections(layout: Layout) -> Iterable[Line2D]:
    """Draws the connections between ancilla qubits and its neighbors.

    Parameters
    ----------
    layout
        The layout to draw the connections of.
    """
    default_params = dict(
        linestyle="--",
    )

    for anc_qubit in layout.anc_qubits:
        anc_coords = layout.param("coords", anc_qubit)
        if len(anc_coords) != 2:
            raise ValueError(
                "Coordinates must be 2D to be plotted, "
                f"but {len(anc_coords)}D were given for qubit {anc_qubit}."
            )

        metaparams = layout.param("metaparams", anc_qubit)
        line_params = deepcopy(default_params)
        stab_type = layout.param("stab_type", anc_qubit)
        if stab_type == "z_type":
            line_params["color"] = COLORS["blue"]
        elif stab_type == "x_type":
            line_params["color"] = COLORS["red"]
        else:
            line_params["color"] = COLORS["green"]
        if isinstance(metaparams, dict):
            custom_params = metaparams.get("line", {})
            line_params.update(custom_params)

        for nbr in layout.get_neighbors(anc_qubit):
            nbr_coords = layout.param("coords", nbr)
            line_coords = (anc_coords, nbr_coords)

            yield get_line(line_coords, **line_params)


def qubit_artists(layout: Layout) -> Iterable[Circle]:
    """Draws the qubits of a layout.

    Parameters
    ----------
    layout
        The layout to draw the qubits of.
    """
    default_radius = 0.3
    default_params = dict(edgecolor="black")

    for qubit in layout.qubits:
        coords = layout.param("coords", qubit)
        if len(coords) != 2:
            raise ValueError(
                "Coordinates must be 2D to be plotted, "
                f"but {len(coords)}D were given for qubit {qubit}."
            )

        metaparams = layout.param("metaparams", qubit)
        radius = deepcopy(default_radius)
        circle_params = deepcopy(default_params)
        if layout.param("role", qubit) == "data":
            circle_params["facecolor"] = "white"
        else:
            stab_type = layout.param("stab_type", qubit)
            if stab_type == "z_type":
                circle_params["facecolor"] = COLORS["blue"]
            elif stab_type == "x_type":
                circle_params["facecolor"] = COLORS["red"]
            else:
                circle_params["facecolor"] = COLORS["green"]

        if isinstance(metaparams, dict):
            custom_params = metaparams.get("circle", {})
            circle_params.update(custom_params)
            radius = circle_params.pop("radius", default_radius)

        yield get_circle(coords, radius, **circle_params)


def logical_artists(layout: Layout) -> Iterable[Wedge]:
    """Draws the logical support of a layout.

    Parameters
    ----------
    layout
        The layout to draw the qubits of.
    """
    default_radius = 0.3
    width = 0.1
    default_params = dict(edgecolor="none")

    logical_qubits = layout.logical_qubits
    if len(logical_qubits) == 0:
        return
    angle = 360 / len(logical_qubits)
    support = {
        "z": {l: layout.logical_param("log_z", l) for l in logical_qubits},
        "x": {l: layout.logical_param("log_x", l) for l in logical_qubits},
    }
    support_x = [q for supp in support["x"].values() for q in supp]

    for k, logical_qubit in enumerate(logical_qubits):
        for pauli in ["z", "x"]:
            for qubit in support[pauli][logical_qubit]:
                coords = layout.param("coords", qubit)
                if len(coords) != 2:
                    raise ValueError(
                        "Coordinates must be 2D to be plotted, "
                        f"but {len(coords)}D were given for qubit {qubit}."
                    )

                wedge_params = deepcopy(default_params)
                wedge_params["facecolor"] = COLORS["red"]
                radius = default_radius + width
                if pauli == "z":
                    wedge_params["facecolor"] = COLORS["blue"]
                    if qubit in support_x:
                        radius = default_radius + 2 * width

                yield get_wedge(
                    coords,
                    radius,
                    theta1=k * angle,
                    theta2=(k + 1) * angle,
                    width=width,
                    **wedge_params,
                )


def patch_artists(layout: Layout) -> Iterable[Polygon]:
    """Draws the stabilizer patches of a layout.

    Parameters
    ----------
    layout
        The layout to draw the patches of.
    """
    default_params = dict(edgecolor="black")
    for anc_qubit in layout.anc_qubits:
        anc_coords = layout.param("coords", anc_qubit)
        if len(anc_coords) != 2:
            raise ValueError(
                "Coordinates must be 2D to be plotted, "
                f"but {len(anc_coords)}D were given for qubit {anc_qubit}."
            )

        neigbors = layout.get_neighbors(anc_qubit)
        coords = [layout.param("coords", nbr) for nbr in neigbors]

        # if the ancilla is only connected to two other data qubits,
        # then the ancilla is one of the vertices of the stabilizer patch.
        if len(neigbors) == 2:
            coords.append(anc_coords)

        # sort the coordinates so that a correct polygon is drawn.
        patch_coords = clockwise_sort(coords)

        metaparams = layout.param("metaparams", anc_qubit)
        patch_params = deepcopy(default_params)
        stab_type = layout.param("stab_type", anc_qubit)
        if stab_type == "z_type":
            patch_params["facecolor"] = COLORS["blue_light"]
        elif stab_type == "x_type":
            patch_params["facecolor"] = COLORS["red_light"]
        else:
            patch_params["facecolor"] = COLORS["green_light"]

        if isinstance(metaparams, dict):
            custom_params = metaparams.get("patch", {})
            patch_params.update(custom_params)
        yield get_patch(patch_coords, **patch_params)


def get_coord_range(layout: Layout) -> tuple[CoordRange, CoordRange]:
    """Returns the range for the X and Y coordinates in the Layout.

    Parameters
    ----------
    layout
        Layout of which to compute the coordinate range.

    Returns
    -------
    [(x_min, x_max), (y_min, y_max)].
    """
    list_coords = [layout.param("coords", qubit) for qubit in layout.qubits]
    for coords in list_coords:
        if len(coords) != 2:
            raise ValueError(
                "Coordinates must be 2D to be plotted, "
                f"but {len(coords)}D were given."
            )
    x_coords, y_coords = zip(*list_coords)

    x_range: CoordRange = (min(x_coords), max(x_coords))
    y_range: CoordRange = (min(y_coords), max(y_coords))
    return x_range, y_range


def plot(
    ax: Axes,
    *layouts: Layout,
    add_labels: bool = True,
    add_patches: bool = True,
    add_connections: bool = True,
    add_logicals: bool = True,
    pad: float = 1,
    stim_orientation: bool = True,
    label_fontsize: float | int = 11,
) -> Axes:
    """Plots a layout.

    Parameters
    ----------
    ax
        The axis to plot the layout on.
    *layouts
        List of layouts to plot.
    add_labels
        Flag to add qubit labels, by default ``True``.
    add_patches
        Flag to plot stabilizer patches, by default ``True``.
    add_connections
        Flag to plot lines indicating the connectivity, by default ``True``.
    add_logicals
        Flag to highlight the logical support on the data qubits, by default ``True``.
    pad
        The padding to the bottom axis, by default ``1``.
    stim_orientation
        Flag to orient the layout and axis as stim does for ``diagram``.
    label_fontsize
        Default font size of the qubit labels. If ``layout`` has information
        about the font size, then this argument is ignored. The purpose
        of this argument is to easily scale down the label size for
        large codes.

    Returns
    -------
    ax
        The figure the layout was plotted on.
    """
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for layout in layouts:
        for artist in qubit_artists(layout):
            ax.add_artist(artist)

        if add_logicals:
            for artist in logical_artists(layout):
                ax.add_artist(artist)

        if add_patches:
            for artist in patch_artists(layout):
                ax.add_artist(artist)

        if add_connections:
            for artist in qubit_connections(layout):
                ax.add_artist(artist)

        if add_labels:
            for artist in qubit_labels(layout, label_fontsize):
                ax.add_artist(artist)

        x_range, y_range = get_coord_range(layout)
        x_min, x_max = min(x_min, x_range[0]), max(x_max, x_range[1])
        y_min, y_max = min(y_min, y_range[0]), max(y_max, y_range[1])

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("$x$ coordinate")
    ax.set_ylabel("$y$ coordinate")
    ax.set_aspect("equal")
    if stim_orientation:
        ax.invert_yaxis()

    return ax

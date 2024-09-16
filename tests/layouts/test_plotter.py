import matplotlib.pyplot as plt

from qec_util.layouts import rot_surf_code
from qec_util.layouts import plot


def test_plot(show_figures):
    _, ax = plt.subplots()
    layout = rot_surf_code(3)

    plot(ax, layout)

    if show_figures:
        plt.show()
    plt.close()

    return

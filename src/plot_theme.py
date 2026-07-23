"""Shared cyberpunk styling for the generated Matplotlib figures."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure


BACKGROUND = "#404040"
NEON_PINK = "#ff2bd6"
NEON_GREEN = "#39ff14"
WHITE = "#ffffff"


def style_figure(fig: Figure) -> None:
    """Apply the dark background and white foreground to a figure."""

    fig.patch.set_facecolor(BACKGROUND)


def style_3d_axis(ax: Axes) -> None:
    """Apply the cyberpunk palette to a Matplotlib 3D axis."""

    ax.set_facecolor(BACKGROUND)
    ax.title.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.zaxis.label.set_color(WHITE)
    ax.tick_params(colors=WHITE)

    grid_rgba = (1.0, 1.0, 1.0, 0.22)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(BACKGROUND)
        axis.pane.set_edgecolor(WHITE)
        axis.pane.set_alpha(1.0)
        axis.line.set_color(WHITE)
        axis._axinfo["grid"]["color"] = grid_rgba
        axis._axinfo["grid"]["linewidth"] = 0.65
        axis._axinfo["tick"]["color"] = WHITE
        axis._axinfo["axisline"]["color"] = WHITE


def style_colorbar(colorbar: Colorbar) -> None:
    """Style a colorbar for use on the dark figure background."""

    colorbar.ax.set_facecolor(BACKGROUND)
    colorbar.ax.tick_params(colors=WHITE)
    colorbar.ax.yaxis.label.set_color(WHITE)
    colorbar.outline.set_edgecolor(WHITE)

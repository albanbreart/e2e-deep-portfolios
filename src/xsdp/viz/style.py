"""Shared plotting style: one visual system for every figure in the report.

The palette is a validated categorical set — the slot order is the
colour-vision-deficiency safety mechanism, not decoration, so series are assigned
slots in a fixed order and the order is never cycled.  Three of the slots fall
below 3:1 contrast on a white page, which is why every line carries a direct
label at its right end rather than relying on a legend swatch alone.

Everything else follows from two rules: the data is the darkest thing on the
page, and no ink is spent on anything that is not data.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Fixed categorical order. Never cycle it; a ninth series folds into "other".
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text, axis labels
GRID = "#e6e5e1"       # recessive grid
RULE = "#c9c8c3"       # axis rule / zero line
MUTED = "#a8a7a2"      # de-emphasised series
SURFACE = "#ffffff"

#: Diverging pair for signed quantities (blue = good, red = bad, gray midpoint).
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"

FIGSIZE = (6.9, 3.9)       # fits a two-column LaTeX page at full width
FIGSIZE_WIDE = (6.9, 3.1)
FIGSIZE_HALF = (3.35, 2.7)


def use_report_style() -> None:
    """Install the report's matplotlib defaults. Call once per script."""
    mpl.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,

        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.axisbelow": True,

        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.grid.axis": "y",

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "xtick.minor.size": 0,
        "xtick.major.width": 0.8,

        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.markersize": 4.5,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
    })


def colour(i: int) -> str:
    """Slot ``i`` of the categorical palette, clamped rather than wrapped."""
    return SERIES[min(i, len(SERIES) - 1)]


def label_line_end(ax, x, y, text: str, colour_: str, *, pad_points: float = 7.0,
                   fontsize: float = 8.0) -> None:
    """Write a series name just past the right end of its line.

    Direct labelling is what discharges the contrast warning on the lighter
    palette slots, and it removes the eye's round trip to a legend box.  The
    gap is specified in **typographic points**, not data units, so it looks the
    same whether the x-axis spans 40 years or 120 basis points.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(pad_points, 0),
        textcoords="offset points", va="center", ha="left",
        fontsize=fontsize, color=colour_, fontweight="medium",
        annotation_clip=False,
    )


def limit_ticks_to_data(ax, lo: float, hi: float, *, axis: str = "x",
                        nbins: int = 7, integer: bool = True) -> None:
    """Place evenly spaced ticks inside the data range after padding the limits.

    Widening a limit to make room for end-labels invites the locator to put a
    tick in the empty margin, which reads as data that is not there.  Filtering
    those out afterwards leaves *uneven* spacing (2010, 2012, 2015, 2017...),
    which is worse; so we re-run the locator on the data range and use what it
    gives, which is regular by construction.
    """
    from matplotlib.ticker import MaxNLocator

    ticks = [t for t in MaxNLocator(nbins=nbins, integer=integer).tick_values(lo, hi)
             if lo - 1e-9 <= t <= hi + 1e-9]
    if axis == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)


def zero_line(ax, *, axis: str = "y") -> None:
    """A hairline at zero -- the reference every return chart needs."""
    if axis == "y":
        ax.axhline(0, color=RULE, lw=0.9, zorder=1)
    else:
        ax.axvline(0, color=RULE, lw=0.9, zorder=1)


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional explanatory second line.

    Both lines are drawn as annotations stacked upward from the axes, rather
    than using ``set_title`` for one and an annotation for the other -- mixing
    the two puts them at independent offsets and they collide.
    """
    base = 20 if subtitle else 6
    ax.annotate(text, xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, base), textcoords="offset points",
                fontsize=9.5, color=INK, fontweight="semibold",
                ha="left", va="bottom", annotation_clip=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8, color=INK_2, ha="left", va="bottom",
                    annotation_clip=False)


def log_return_axis(ax) -> None:
    """Format a log y-axis for growth multiples: labelled decades, no minor clutter.

    Matplotlib's default log axis draws eight unlabelled minor ticks per decade,
    which on a figure this size reads as visual noise attached to the axis.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, path, *, also_png: bool = True) -> None:
    """Write the figure as PDF (for LaTeX) and PNG (for the README)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)

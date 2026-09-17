"""Shared publication figure style for the Rapid Fitting paper.

One source of truth for colors, fonts, and sizes so every figure is
consistent, readable, and colorblind-safe. Import and call set_pub_style()
at the top of any figure/analysis script, and pull method colors from
METHOD_COLORS so a given method has ONE fixed color in every figure.

    from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP
    set_pub_style()
    sns.violinplot(..., palette={"LSQF": METHOD_COLORS["LSQF"],
                                 "NN":   METHOD_COLORS["NN"]})
"""
import matplotlib as mpl

# Okabe-Ito colorblind-safe categorical palette (distinguishable under the
# common forms of color-vision deficiency).
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "gray":       "#666666",
}

# Fixed color per method/series. Use these EVERYWHERE so the reader builds one
# stable color key: LSQF is always blue, the network always orange, etc.
# Distinct hues guarantee no two legend entries share a color.
METHOD_COLORS = {
    "LSQF":             OKABE_ITO["blue"],        # least squares
    "NN":               OKABE_ITO["orange"],      # the network
    "Network":          OKABE_ITO["orange"],
    "network":          OKABE_ITO["orange"],
    "Adam":             OKABE_ITO["green"],
    "TR-CG":            OKABE_ITO["purple"],
    "Trust Region CG":  OKABE_ITO["purple"],
    "AdaHessian":       OKABE_ITO["vermillion"],
    "Raw Data":         OKABE_ITO["gray"],
    "Raw":              OKABE_ITO["gray"],
    "network-Adam":     OKABE_ITO["orange"],
    "network-TR-CG":    OKABE_ITO["green"],
}

# Perceptually-uniform, CVD-friendly sequential colormap for maps / density /
# hexbin (replaces the default viridis where a CVD-optimized map is preferred).
SEQ_CMAP = "cividis"


def set_pub_style():
    """Apply the shared rcParams: legible fonts, clean despined axes, rectangular
    default aspect, vector-friendly text."""
    mpl.rcParams.update({
        # resolution
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # rectangular default (golden-ish), scripts may override per figure
        "figure.figsize": (6.4, 3.6),
        # fonts -- bigger and consistent
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        # clean axes
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.grid": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        # keep text as text in SVG/PDF (selectable, sharp)
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

#!/usr/bin/env python3
"""Build main-text Fig 5: ground-truth parameter recovery vs noise.

Inputs: the E13 fair-baseline study results (e13_results.json, produced by
analysis/e13_pipeline.py). Four estimators x two truth families; the network
line is seed 42, with a translucent min-max bar over seeds {42,43,44} at the
noise levels where all three were trained (family (a), n in {2, 8}).

Amplitude is reported as percent of the fitter range (A in [0, 1.5e-4], the
bounds used by every estimator in the study).

Output: figures/fig_truth_error_noise.png
Usage: python make_fig_truth_error.py [path/to/e13_results.json]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pubstyle import set_pub_style  # noqa: E402

set_pub_style()

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "..", "..", "..", "revision_records", "e13_results.json"),
    os.path.join(HERE, "..", "..", "..", "analysis", "e13_results.json"),
]
path = sys.argv[1] if len(sys.argv) > 1 else next(
    (p for p in CANDIDATES if os.path.isfile(p)), CANDIDATES[0])
D = json.load(open(path))

NOISE = [0, 2, 4, 6, 8]
A_RANGE = 1.5e-4  # fitter bounds for A, shared by all estimators

METHODS = [
    ("lsqf_heuristic", "LSQF (heuristic init)", "#0072B2", "o"),
    ("lsqf_multistart", "LSQF (multistart)", "#56B4E9", "v"),
    ("map", "MAP (empirical prior)", "#CC79A7", "D"),
    ("nn", "network (noise-matched)", "#E69F00", "s"),
]
PANELS = [
    ("A", "Amplitude $A$", "median $|\\Delta A|$ (% of range)", 100.0 / A_RANGE),
    ("w0", "Resonance $\\omega_0$", "median $|\\Delta\\omega_0|$ (kHz)", 1e-3),
    ("Q", "Quality factor $Q$", "median $|\\Delta Q|$", 1.0),
    ("phi", "Phase $\\phi$", "median $|\\Delta\\phi|$ (rad, circular)", 1.0),
]


def series(fam, key, param, scale):
    ns, vs = [], []
    for n in NOISE:
        cell = D[f"{fam}_n{n}"]
        k = "nn_s42" if key == "nn" else key
        if k in cell:
            ns.append(n)
            vs.append(cell[k][param] * scale)
    return ns, vs


fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
for ax, (param, title, ylabel, scale) in zip(axes.ravel(), PANELS):
    for key, label, color, marker in METHODS:
        for fam, ls in (("a", "-"), ("b", "--")):
            ns, vs = series(fam, key, param, scale)
            ax.plot(ns, vs, ls, marker=marker, color=color, markersize=6,
                    label=label if (fam == "a" and param == "A") else None)
        if key == "nn":  # seed min-max bars, family (a), n in {2, 8}
            for n in (2, 8):
                seeds = [D[f"a_n{n}"][f"nn_s{s}"][param] * scale
                         for s in (42, 43, 44)]
                ax.plot([n, n], [min(seeds), max(seeds)], color=color,
                        alpha=0.35, linewidth=7, solid_capstyle="round",
                        zorder=1)
    ax.set_title(title)
    ax.set_xlabel("Noise factor $n$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(NOISE)
    ax.set_ylim(bottom=0)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.text(0.5, 0.005,
         "solid: synthetic truth (a) · dashed: semi-synthetic truth (b) "
         "· thick bars: seed spread (3 seeds) "
         "· MAP on 10k-spectrum subsamples",
         ha="center", fontsize=10, color="0.45")
fig.tight_layout(rect=(0, 0.02, 1, 0.94))

OUT = os.path.abspath(os.path.join(HERE, "..", ".."))
fig.savefig(os.path.join(OUT, "fig_truth_error_noise.png"), dpi=300)
print("wrote fig_truth_error_noise.png from", os.path.relpath(path, OUT))

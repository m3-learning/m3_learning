#!/usr/bin/env python3
"""Build main-text Fig 4b: reconstruction MSE and out-of-range fraction vs noise.

Inputs: the E4/E5 noise-sweep summary (phase0_E4_E5_noise_sweep.csv). Both
panels compare the deployed LSQF refit against the noise-matched Adam network
(the models the noise study trains at every level):

  left  -- median per-spectrum reconstruction MSE against the noisy input;
  right -- fraction of parameters out of the fitter-configured ranges,
           pooled over the four SHO parameters (phase wrapped), i.e. the
           mean of the per-parameter out-of-range fractions.

Output: figures/fig_accuracy_noise.png
Usage: python make_fig_accuracy_noise.py [path/to/phase0_E4_E5_noise_sweep.csv]
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pubstyle import set_pub_style, METHOD_COLORS  # noqa: E402

set_pub_style()

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "..", "..", "..", "revision_records",
                 "phase0_E4_E5_noise_sweep.csv"),
    os.path.join(HERE, "..", "..", "..", "analysis",
                 "phase0_E4_E5_noise_sweep.csv"),
]
path = sys.argv[1] if len(sys.argv) > 1 else next(
    (p for p in CANDIDATES if os.path.isfile(p)), CANDIDATES[0])
rows = list(csv.DictReader(open(path)))


def pooled_oor(r):
    return (float(r["oor_A"]) + float(r["oor_w0"]) + float(r["oor_Q"])
            + float(r["oor_phi_wrapped"])) / 4.0


SERIES = [("LSQF", "LSQF", METHOD_COLORS["LSQF"]),
          ("NN_matched_Adam", "NN", METHOD_COLORS["NN"])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2))
for method, label, color in SERIES:
    sel = sorted((int(r["noise"]), r) for r in rows if r["method"] == method)
    ns = [n for n, _ in sel]
    ax1.plot(ns, [float(r["medMSE_vs_input"]) for _, r in sel],
             "-o", color=color, label=label)
    ax2.plot(ns, [pooled_oor(r) for _, r in sel], "-o", color=color,
             label=label)

ax1.set_title("Reconstruction MSE vs noise")
ax1.set_xlabel("Noise factor")
ax1.set_ylabel("Median reconstruction MSE")
ax1.legend()
ax2.set_title("Unphysical parameters vs noise")
ax2.set_xlabel("Noise factor")
ax2.set_ylabel("Fraction of parameters out of range")
ax2.set_ylim(0, 1)
ax2.legend(loc="upper left")
for ax in (ax1, ax2):
    ax.set_xticks(range(0, 9))
fig.tight_layout()

OUT = os.path.abspath(os.path.join(HERE, "..", ".."))
fig.savefig(os.path.join(OUT, "fig_accuracy_noise.png"), dpi=300)
print("wrote fig_accuracy_noise.png from", os.path.relpath(path, OUT))

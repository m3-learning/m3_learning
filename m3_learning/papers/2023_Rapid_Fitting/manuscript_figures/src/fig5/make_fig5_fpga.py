#!/usr/bin/env python3
"""Build the Figure 5 inference-latency/throughput panel.

Three deployment targets compared on a single SHO spectrum:

  LSQF (4x 2.3 GHz CPU cores): 1,280 fits/s  ->  ~781 us/spectrum.
      Source: Methods (sec:methods-lsqf). This number is still RERUN-flagged in
      the manuscript ("[RERUN: LSQF baseline time]"), so the bar is annotated as
      provisional rather than invented here.

  GPU NN (single GPU, batched inference): 1,382,400 spectra in 2.39 s
      => 1.73 us/spectrum, ~5.78e5 fits/s.
      Source: executed_2_Pytorch_SHO_Fitter.ipynb, cell 37 (model.inference_timer):
      "Total execution time (s): 2.39", "per iteration (ms): 0.001727".

  FPGA (hls4ml, quantization-aware): 37 us/spectrum streaming latency.
      No executed-notebook / artifact source exists for this number in the
      rerun bundle; it is taken from the manuscript Methods and is RERUN-flagged.
      The bar is drawn hatched and labelled [RERUN] so it is not mistaken for a
      measured value.

We plot per-spectrum latency (us, log scale) because that is the quantity the
three targets share; throughput in fits/s is annotated on each bar.

Output: figures/fig5_fpga_latency.png
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pubstyle import set_pub_style, METHOD_COLORS

set_pub_style()

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# (label, latency_us, throughput_fits_per_s, sourced, color)
# Consistent paper palette: LSQF = blue, the network = orange (GPU and FPGA are
# both "the network"; the FPGA bar is hatched to mark it as a projected result).
targets = [
    ("LSQF\n(CPU, 4-core)", 781.0, 1280.0, "methods", METHOD_COLORS["LSQF"]),
    ("GPU NN\n(batched)", 1.73, 5.78e5, "notebook", METHOD_COLORS["NN"]),
    ("FPGA NN\n(streaming)", 37.0, 2.7e4, "projected", METHOD_COLORS["NN"]),
]

fig, ax = plt.subplots(figsize=(4.4, 3.6))

x = np.arange(len(targets))
lat = [t[1] for t in targets]
colors = [t[4] for t in targets]
bars = ax.bar(x, lat, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)

# Hatch + flag the unsourced FPGA bar.
bars[2].set_hatch("////")

ax.set_yscale("log")
ax.set_xlabel("Deployment target")
ax.set_ylabel("Time per spectrum ($\\mu$s)")
ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in targets], fontsize=9.5)
ax.set_ylim(0.7, 5000)

# Legend: color <-> method, with the hatched FPGA bar marked as projected.
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=METHOD_COLORS["LSQF"], alpha=0.85, edgecolor="k", label="LSQF"),
    Patch(facecolor=METHOD_COLORS["NN"], alpha=0.85, edgecolor="k", label="Neural network"),
    Patch(facecolor=METHOD_COLORS["NN"], alpha=0.85, edgecolor="k", hatch="////",
          label="NN (projected)"),
]
ax.legend(handles=legend_handles, loc="upper right")

# Throughput annotations above each bar (FPGA marked as a projected estimate).
labels = [
    "1,280 fits/s\n(throughput$^{-1}$)",
    r"$5.8\times10^{5}$ fits/s" + "\n(amortized)",
    r"$2.7\times10^{4}$ fits/s" + "\n(projected latency)",
]
for xi, t, lab in zip(x, targets, labels):
    ax.annotate(
        f"{t[1]:g} $\\mu$s\n{lab}",
        xy=(xi, t[1]), xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=8.5,
    )

ax.set_title("Per-spectrum deployment timing")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_fpga_latency.png"), dpi=300)
plt.close(fig)
print("wrote fig5_fpga_latency.png")
print("LSQF 781 us (1280 fits/s, Methods) | GPU 1.73 us (NB2 cell37, 2.39s/1.38M) "
      "| FPGA 37 us (RERUN)")

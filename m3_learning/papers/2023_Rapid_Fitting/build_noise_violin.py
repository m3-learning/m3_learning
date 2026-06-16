"""Figure_noise_violin: split-violin of the 4 SCALED SHO params (A, omega_0, Q, phi),
LSQF vs network, for noise n=0,4,7. Network = per-noise best Adam model (nb4
find_best_model), same source as p6a/p6b. Styling matches the fixed
violin_plot_comparison. NO retraining.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn, find_best_model
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

DATA = "Data/data_raw.h5"
LSQF_C, NN_C = "#0072B2", "#E69F00"
LABELS = ["A", "ω", "Q", "φ"]   # A, omega, Q, phi (match violin_plot_comparison)
NOISES = [0, 4, 7]

set_style("printing"); random_seed(seed=42)
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP
set_pub_style()
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)

basepath = sorted(glob.glob("Trained Models/SHO Fitter/*_nn_benchmarks_noise"))[-1]
results = find_best_model(basepath, "Batch_Trainging_SpeedTest.csv")

def load_adam(noise):
    ck = basepath + "/" + results[(noise, "Adam")]['filename'].split("//")[-1]
    fitter = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4,
                                dataset.SHO_scaler, ComplexPostProcessor(dataset))
    m = Model(fitter, dataset, training=False, model_basename="SHO_Fitter_original_data")
    m.load(ck)
    return m

state = {"fitter": "LSQF", "raw_format": "complex", "resampled": True, "scaled": True,
         "output_shape": "index", "measurement_state": "all"}

frames = {}
counts = {}
for n in NOISES:
    st = dict(state); st["noise"] = n
    dataset.set_attributes(**st)
    model = load_adam(n)
    X, _ = dataset.NN_data()
    _, _, nn_phys = model.predict(X)
    nn_scaled = dataset.SHO_scaler.transform(np.asarray(nn_phys))   # NN scaled (violin recipe)
    dataset.set_attributes(**st)
    lsqf_scaled = dataset.SHO_fit_results().reshape(-1, 4)          # LSQF scaled (violin 'true')
    counts[n] = X.shape[0]
    df = pd.DataFrame()
    for arr, meth in [(lsqf_scaled, "LSQF"), (nn_scaled, "NN")]:
        for i, lab in enumerate(LABELS):
            df = pd.concat((df, pd.DataFrame({
                "value": arr[:, i],
                "parameter": np.repeat(lab, arr.shape[0]),
                "method": np.repeat(meth, arr.shape[0]),
            })), ignore_index=True)
    frames[n] = df
    print(f"noise {n}: best Adam model = {results[(n,'Adam')]['filename'].split('//')[-1]}  "
          f"n_spectra = {X.shape[0]}", flush=True)

# 3 panels, shared y
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, n in zip(axes, NOISES):
    sns.violinplot(data=frames[n], x="parameter", y="value", hue="method", split=True,
                   inner="quart", cut=0, density_norm="width", linewidth=0.8,
                   palette={"LSQF": LSQF_C, "NN": NN_C}, ax=ax)
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlabel(""); ax.set_title(f"noise factor {n}", fontsize=11)
    ax.set_ylim(-5, 5)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
axes[0].set_ylabel("Scaled SHO parameter")
# one shared legend
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LSQF_C, NN_C)]
fig.legend(handles, ["LSQF", "NN"], frameon=False, loc="upper right",
           bbox_to_anchor=(0.995, 0.99), ncol=2)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("./Figures/Figure_noise_violin.png", dpi=300, bbox_inches="tight")
fig.savefig("./Figures/Figure_noise_violin.svg", bbox_inches="tight")
plt.close(fig)
print(f"\nUsed noise levels: {NOISES}")
print("spectra per level:", {n: counts[n] for n in NOISES})
print("Wrote ./Figures/Figure_noise_violin.png and .svg")
print("DONE noise_violin")

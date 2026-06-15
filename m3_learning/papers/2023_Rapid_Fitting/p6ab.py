"""P6a: Fig-3 noise histograms (n=4, n=7), rows = LSQF / network-Adam / network-TR-CG,
consistent palette. P6b: SI switching maps (n=7) with 'SGD'->'TR-CG' label.
Models = per-(noise,optimizer) best from the 2_5 benchmark (nb4 find_best_model recipe).
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.be.viz import Viz
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn, find_best_model
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

LSQF_C, NN_C, TR_C = "#0072B2", "#E69F00", "#009E73"
LABELS = ["A", r"$\omega_0$", "Q", r"$\varphi$"]
RANGES = [(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)]
DATA = "Data/data_raw.h5"

set_style("printing"); random_seed(seed=42)
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP
set_pub_style()
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
printing = printer(basepath="./Figures/")
BE_viz = Viz(dataset, printing, verbose=False, SHO_ranges=RANGES,
             image_scalebar=[2000, 500, "nm", "br"])

# nb4 recipe: best per-(noise,optimizer) model from the 2_5 benchmark
basepath = sorted(glob.glob("Trained Models/SHO Fitter/*_nn_benchmarks_noise"))[-1]
results = find_best_model(basepath, "Batch_Trainging_SpeedTest.csv")
print("benchmark:", basepath)

def load_nn_model(ckpt):
    fitter = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4,
                                dataset.SHO_scaler, ComplexPostProcessor(dataset))
    m = Model(fitter, dataset, training=False, model_basename="SHO_Fitter_original_data")
    m.load(ckpt)
    return m

def models_for(noise):
    a = basepath + "/" + results[(noise, "Adam")]['filename'].split("//")[-1]
    t = basepath + "/" + results[(noise, "Trust Region CG")]['filename'].split("//")[-1]
    return load_nn_model(a), load_nn_model(t)

def get_params(noise, model_adam, model_tr):
    st = {"resampled": True, "raw_format": "complex", "fitter": "LSQF", "scaled": False,
          "output_shape": "index", "measurement_state": "all", "resampled_bins": 165,
          "LSQF_phase_shift": np.pi / 2, "NN_phase_shift": None, "noise": noise}
    dataset.set_attributes(**st)
    lsqf = np.asarray(dataset.SHO_fit_results(state=st)).reshape(-1, 4)
    X, _ = dataset.NN_data()
    adam = np.asarray(dataset.SHO_fit_results(model=model_adam, phase_shift=np.pi / 2, X_data=X)).reshape(-1, 4)
    tr = np.asarray(dataset.SHO_fit_results(model=model_tr, phase_shift=np.pi / 2, X_data=X)).reshape(-1, 4)
    return lsqf, adam, tr

# ---------- P6a: histograms for noise 4 and 7 ----------
for noise in (4, 7):
    ma, mt = models_for(noise)
    lsqf, adam, tr = get_params(noise, ma, mt)
    rows = [("LSQF", lsqf, LSQF_C), ("network – Adam", adam, NN_C), ("network – TR-CG", tr, TR_C)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for r, (rlab, P, c) in enumerate(rows):
        for j in range(4):
            ax = axes[r, j]
            lo, hi = RANGES[j]
            ax.hist(np.clip(P[:, j], lo, hi), bins=120, range=(lo, hi), color=c)
            if r == 0:
                ax.set_title(LABELS[j])
            if j == 0:
                ax.set_ylabel(rlab + "\ncount", fontsize=11)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
    fig.suptitle(f"SHO parameter histograms — noise factor {noise}  "
                 f"(rows: LSQF / network-Adam / network-TR-CG)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"./Figures/Figure_3_histograms_noise{noise}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"./Figures/Figure_3_histograms_noise{noise}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote ./Figures/Figure_3_histograms_noise{noise}.png and .svg", flush=True)

# ---------- P6b: switching maps n=7 ----------
noise = 7
ma, mt = models_for(noise)
lsqf, adam, tr = get_params(noise, ma, mt)
BE_viz.SHO_switching_maps_test(
    [lsqf, adam, tr],
    filename=f"Figure_5_47_switching_maps_comparison_{noise}_noise",
    labels=["LSQF", "Adam", "TR-CG"],
)
print(f"Wrote ./Figures/Figure_5_47_switching_maps_comparison_{noise}_noise.png and .svg", flush=True)
print("DONE P6ab", flush=True)

"""Accuracy/robustness vs noise — PER-NOISE best Adam models (find_best_model),
matching the noise violin / p6. Panel (a): median per-spectrum reconstruction MSE.
Panel (b): FRACTION of recovered SHO parameters strictly outside the physically
admissible ranges (A,(0,1.5e-4); omega_0,(1.31e6,1.33e6); Q,(-300,0); phi wrapped
to (-pi,pi) -> never out), for LSQF and the network, n=0..8. Overwrites
./Figures/Figure_accuracy_vs_noise.{png,svg}.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn, find_best_model
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP

DATA = "Data/data_raw.h5"
LSQF_C, NN_C = METHOD_COLORS["LSQF"], METHOD_COLORS["NN"]
RANGES = [(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)]

set_style("printing"); random_seed(seed=42); set_pub_style()
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

def to_arr(lst):
    return np.rollaxis(np.array(lst), 0, 3)

def frac_out_of_range(P):
    """fraction of individual parameter values strictly outside RANGES; phi is
    wrapped to (-pi,pi] first so it is never counted as unphysical (it is circular)."""
    P = np.asarray(P).astype(float).copy()
    P[:, 3] = np.angle(np.exp(1j * P[:, 3]))      # wrap phi
    out = 0
    for i, (lo, hi) in enumerate(RANGES):
        out += int(np.sum((P[:, i] < lo) | (P[:, i] > hi)))
    return out / P.size                            # P.size = N * 4

base = {"fitter": "LSQF", "raw_format": "complex", "resampled": True,
        "scaled": True, "output_shape": "index", "measurement_state": "all"}

rows = []
for k in range(9):
    st = dict(base); st["noise"] = k
    dataset.set_attributes(**st)
    model = load_adam(k)                      # PER-NOISE best Adam model
    X, _ = dataset.NN_data()
    _, _, nn_phys = model.predict(X)
    nn_phys = np.asarray(nn_phys)
    dataset.set_attributes(**st); dataset.scaled = False
    lsqf_phys = dataset.SHO_fit_results().reshape(-1, 4)
    dataset.scaled = True
    true_list, _ = dataset.raw_spectra(frequency=True)
    nn_recon, _ = dataset.raw_spectra(fit_results=nn_phys, frequency=True)
    lsqf_recon, _ = dataset.raw_spectra(fit_results=lsqf_phys, frequency=True)
    t = to_arr(true_list)
    mse_nn = float(np.median(Model.MSE(t, to_arr(nn_recon))))
    mse_lsqf = float(np.median(Model.MSE(t, to_arr(lsqf_recon))))
    oor_lsqf = frac_out_of_range(lsqf_phys)
    oor_nn = frac_out_of_range(nn_phys)
    rows.append((k, mse_lsqf, mse_nn, oor_lsqf, oor_nn))
    print(f"noise {k}: medMSE LSQF={mse_lsqf:.5f} NN={mse_nn:.5f} | "
          f"fracOOR LSQF={oor_lsqf:.5f} NN={oor_nn:.6f}", flush=True)

rows = np.array(rows)
print("\nPER-NOISE table (noise, medMSE_LSQF, medMSE_NN, fracOOR_LSQF, fracOOR_NN):")
for r in rows:
    print(f"  {int(r[0])}  {r[1]:.5f}  {r[2]:.5f}  {r[3]:.5f}  {r[4]:.6f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
a1.plot(rows[:, 0], rows[:, 1], "-o", color=LSQF_C, label="LSQF")
a1.plot(rows[:, 0], rows[:, 2], "-o", color=NN_C, label="NN")
a1.set_xlabel("Noise factor"); a1.set_ylabel("Median reconstruction MSE")
a1.set_title("Reconstruction MSE vs noise"); a1.legend(frameon=False)

a2.plot(rows[:, 0], rows[:, 3], "-o", color=LSQF_C, label="LSQF")
a2.plot(rows[:, 0], rows[:, 4], "-o", color=NN_C, label="NN")
a2.set_xlabel("Noise factor"); a2.set_ylabel("Fraction of parameters out of range")
a2.set_ylim(0, 1); a2.set_title("Unphysical parameters vs noise"); a2.legend(frameon=False)

fig.tight_layout()
fig.savefig("./Figures/Figure_accuracy_vs_noise.png", dpi=300, bbox_inches="tight")
fig.savefig("./Figures/Figure_accuracy_vs_noise.svg", bbox_inches="tight")
plt.close(fig)
print("\nWrote ./Figures/Figure_accuracy_vs_noise.png and .svg (panel b = fraction out of range)")
print("DONE")

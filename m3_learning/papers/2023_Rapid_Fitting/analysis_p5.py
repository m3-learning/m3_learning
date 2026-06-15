"""Priority 5: accuracy/robustness vs noise factor (n=0..8).

For each noise level compute median per-spectrum reconstruction MSE for LSQF and the
network, plus the spread (std) of the scaled SHO parameters. Plot metric vs noise.
Also re-reports phi agreement with circular (wrapped) handling for the manuscript.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

LSQF_C, NN_C = "#3B75AF", "#EF8636"
DATA = "Data/data_raw.h5"
CKPT = ("Trained Models/SHO Fitter/"
        "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.03403592322973526.pth")

set_style("printing"); random_seed(seed=42)
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
post = ComplexPostProcessor(dataset)
model_ = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4, dataset.SHO_scaler, post)
model = Model(model_, dataset, training=True, model_basename="SHO_Fitter_original_data")
model.load(CKPT)

def to_arr(lst):
    return np.rollaxis(np.array(lst), 0, 3)

base = {"fitter": "LSQF", "raw_format": "complex", "resampled": True,
        "scaled": True, "output_shape": "index", "measurement_state": "all"}

rows = []
for k in range(9):
    st = dict(base); st["noise"] = k
    dataset.set_attributes(**st)
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
    # param spread: mean over the 4 params of the scaled-param std
    nn_scaled = dataset.SHO_scaler.transform(nn_phys)
    lsqf_scaled = dataset.SHO_scaler.transform(lsqf_phys)
    spread_nn = float(np.mean(np.std(nn_scaled, axis=0)))
    spread_lsqf = float(np.mean(np.std(lsqf_scaled, axis=0)))
    rows.append((k, mse_lsqf, mse_nn, spread_lsqf, spread_nn))
    print(f"noise {k}: median MSE  LSQF={mse_lsqf:.5f}  NN={mse_nn:.5f} | "
          f"mean param std  LSQF={spread_lsqf:.4f}  NN={spread_nn:.4f}", flush=True)

rows = np.array(rows)
print("\nPRIORITY 5 table (noise, medMSE_LSQF, medMSE_NN, spread_LSQF, spread_NN):")
for r in rows:
    print(f"  {int(r[0])}  {r[1]:.5f}  {r[2]:.5f}  {r[3]:.4f}  {r[4]:.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
a1.plot(rows[:, 0], rows[:, 1], "-o", color=LSQF_C, label="LSQF")
a1.plot(rows[:, 0], rows[:, 2], "-o", color=NN_C, label="NN")
a1.set_xlabel("noise factor"); a1.set_ylabel("median reconstruction MSE")
a1.set_title("Reconstruction MSE vs noise"); a1.legend(frameon=False)
a2.plot(rows[:, 0], rows[:, 3], "-o", color=LSQF_C, label="LSQF")
a2.plot(rows[:, 0], rows[:, 4], "-o", color=NN_C, label="NN")
a2.set_xlabel("noise factor"); a2.set_ylabel("mean scaled-param std")
a2.set_title("Parameter spread vs noise"); a2.legend(frameon=False)
fig.tight_layout()
fig.savefig("./Figures/Figure_accuracy_vs_noise.png", dpi=300, bbox_inches="tight")
fig.savefig("./Figures/Figure_accuracy_vs_noise.svg", bbox_inches="tight")
plt.close(fig)
print("\nWrote ./Figures/Figure_accuracy_vs_noise.png and .svg")
print("DONE P5")

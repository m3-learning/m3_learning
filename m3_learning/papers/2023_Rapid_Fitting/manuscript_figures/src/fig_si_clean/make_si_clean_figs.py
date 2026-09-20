#!/usr/bin/env python3
"""Build SI Figs S1 and S2: clean-data parameter agreement and per-spectrum
reconstruction-MSE distributions, NN versus LSQF, over all 1,382,400 measured
spectra.

Inputs:
  - data_raw.h5 (Zenodo 10.5281/zenodo.7774788): raw spectra + cached LSQF
    SHO fits;
  - the released clean Adam checkpoint of the Multiscale1DFitter
    (SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.034*.pth).

The scalers replicate the training pipeline exactly: a global per-channel
mean/std raw scaler, and a StandardScaler over all cached LSQF parameter rows
with the phase channel left unscaled. Reconstruction MSE is computed in the
scaled space as the mean over all scaled elements of both channels (the
training-loss convention), the quantity reported throughout the paper.

Sanity anchors printed at the end (values from the manuscript): LSQF median
0.0342, NN median 0.0337, NN lower on 69.7% of spectra, A Pearson r = 0.996.

Usage: python make_si_clean_figs.py [data_raw.h5] [checkpoint.pth]
"""
import glob
import os
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/Desktop/Projects/m3_learning/m3_learning/src"))
from pubstyle import set_pub_style, METHOD_COLORS  # noqa: E402

set_pub_style()
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", ".."))
H5 = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Desktop/Projects/m3_learning/m3_learning/papers/2023_Rapid_Fitting/Data/data_raw.h5")
CKPT = sys.argv[2] if len(sys.argv) > 2 else (glob.glob(os.path.expanduser(
    "~/Downloads/rapid_fitting_artifacts/Trained Models/SHO Fitter/"
    "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.034*.pth"))
    + [None])[0]
assert CKPT and os.path.isfile(CKPT), "clean Adam checkpoint not found; pass its path"

# ---------------- data ----------------
import h5py  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
f = h5py.File(H5, "r")
fit = f["Measurement_000/Channel_000/Raw_Data-SHO_Fit_000/Fit"][()]
P_lsqf = np.stack([fit["Amplitude [V]"], fit["Frequency [Hz]"],
                   fit["Quality Factor"], fit["Phase [rad]"]], -1
                  ).reshape(-1, 4).astype(np.float64)
sv = f["Measurement_000/Channel_000/Spectroscopic_Values"][()]
w = np.sort(np.unique(sv[0]).astype(np.float64))
raw = f["Measurement_000/Channel_000/Raw_Data"][()].reshape(-1, 165).astype(np.complex64)
log(f"raw {raw.shape}, LSQF params {P_lsqf.shape}, grid {len(w)} bins")

mu_re, sd_re = float(raw.real.mean()), float(raw.real.std())
mu_im, sd_im = float(raw.imag.mean()), float(raw.imag.std())

# ---------------- model ----------------
import torch  # noqa: E402

torch.set_num_threads(8)
from m3_learning.be.nn import SHO_fit_func_nn  # noqa: E402
from m3_learning.nn.Fitter1D.Fitter1D import (  # noqa: E402
    ComplexPostProcessor, Multiscale1DFitter)


class _Chan:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std


class _Raw:
    def __init__(self):
        self.real_scaler = _Chan(mu_re, sd_re)
        self.imag_scaler = _Chan(mu_im, sd_im)


class _Latent:
    pass


class _DS:
    pass


ls = _Latent()
ls.mean_ = P_lsqf.mean(0)
ls.var_ = P_lsqf.var(0)
ls.mean_[3] = 0.0
ls.var_[3] = 1.0
ds = _DS()
ds.raw_data_scaler = _Raw()
ds.SHO_scaler = ls
ds.frequency_bin = w
ds.noise = 0
post = ComplexPostProcessor(ds)
model_ = Multiscale1DFitter(SHO_fit_func_nn, w, 2, 4, ls, post)
state = torch.load(CKPT, map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model_.load_state_dict(state)
model_.train()  # no dropout/BN: deterministic; predict() has a shape bug
log("checkpoint loaded: " + os.path.basename(CKPT))

Xs = np.stack([(raw.real - mu_re) / sd_re, (raw.imag - mu_im) / sd_im],
              -1).astype(np.float32)
preds = []
with torch.no_grad():
    XT = torch.tensor(Xs)
    for i in range(0, len(XT), 8192):
        _, up = model_(XT[i:i + 8192])
        preds.append(up.cpu().numpy())
P_nn = np.concatenate(preds).astype(np.float64)
del preds, XT, Xs
# Canonicalize onto the A > 0 branch of the exact (A, phi) -> (-A, phi + pi)
# degeneracy of the SHO response (Methods, "Ground-truth recovery study");
# the bounded fitter is constrained to that branch already.
neg = P_nn[:, 0] < 0
P_nn[neg, 0] = -P_nn[neg, 0]
P_nn[neg, 3] = np.angle(np.exp(1j * (P_nn[neg, 3] + np.pi)))
log(f"NN params {P_nn.shape}; canonicalized {neg.mean()*100:.1f}% negative-A outputs")


def sho_np(p, w):
    A, w0, Q, phi = (np.asarray(p)[:, i][:, None] for i in range(4))
    return A * np.exp(1j * phi) * w0**2 / (w**2 - 1j * w * w0 / Q - w0**2)


def scaled_mse(params):
    out = np.empty(len(params))
    for i in range(0, len(params), 100000):
        rec = sho_np(params[i:i + 100000], w)
        seg = raw[i:i + 100000]
        # mean over ALL scaled elements (both channels), the training-loss
        # convention (torch MSELoss) used for every MSE quoted in the paper
        out[i:i + 100000] = ((((rec.real - seg.real) / sd_re) ** 2).mean(1)
                             + (((rec.imag - seg.imag) / sd_im) ** 2).mean(1)) / 2.0
    return out


mse_lsqf = scaled_mse(P_lsqf)
mse_nn = scaled_mse(P_nn)
log("per-spectrum MSEs computed")

# ---------------- Fig S2: MSE distributions ----------------
finite = np.isfinite(mse_lsqf) & np.isfinite(mse_nn)
ml, mn = mse_lsqf[finite], mse_nn[finite]
hi = np.quantile(np.concatenate([ml, mn]), 0.999)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2))
for vals, label, color in ((ml, "LSQF", METHOD_COLORS["LSQF"]),
                           (mn, "NN", METHOD_COLORS["NN"])):
    xs = np.sort(vals)
    ax1.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, label=label)
    ax2.hist(vals, bins=100, range=(0, hi), alpha=0.55, color=color,
             label=label)
ax1.set_xlim(0, hi)
ax1.set_title("CDF")
ax1.set_xlabel("per-spectrum reconstruction MSE")
ax1.set_ylabel("CDF")
ax1.legend()
ax2.set_title("Histogram")
ax2.set_xlabel("per-spectrum reconstruction MSE")
ax2.set_ylabel("count")
ax2.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_mse_dist.png"), dpi=300)
plt.close(fig)
log("wrote fig_mse_dist.png")

# ---------------- Fig S1: parameter agreement hexbins ----------------
ok = np.isfinite(P_lsqf).all(1) & np.isfinite(P_nn).all(1)
fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator

for ax, (idx, name) in zip(axes, [(0, "A"), (1, "omega_0"), (2, "Q")]):
    x, y = P_lsqf[ok, idx], P_nn[ok, idx]
    r = np.corrcoef(x, y)[0, 1]
    r2 = r2_score(x, y)  # agreement with the identity line y = x
    lo = min(np.quantile(x, 0.0005), np.quantile(y, 0.0005))
    hi2 = max(np.quantile(x, 0.9995), np.quantile(y, 0.9995))
    pad = 0.04 * (hi2 - lo)
    keep = (x >= lo) & (x <= hi2) & (y >= lo) & (y <= hi2)
    hb = ax.hexbin(x[keep], y[keep], gridsize=60, bins="log", cmap="cividis",
                   mincnt=1)
    ax.plot([lo, hi2], [lo, hi2], "r--", label="y = x")
    ax.set_xlim(lo - pad, hi2 + pad)
    ax.set_ylim(lo - pad, hi2 + pad)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set_title(f"{name}   (Pearson r={r:.3f}, R$^2$={max(r2, 0):.3f})",
                 fontsize=13)
    ax.set_xlabel(f"LSQF  {name}")
    ax.set_ylabel(f"NN  {name}")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="log10(count)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_param_scatter.png"), dpi=300)
plt.close(fig)
log("wrote fig_param_scatter.png")

# ---------------- sanity anchors ----------------
print("\n=== sanity anchors (manuscript values in brackets) ===")
print(f"LSQF median MSE : {np.median(ml):.4f}  [0.0342]")
print(f"NN   median MSE : {np.median(mn):.4f}  [0.0337]")
print(f"NN lower on     : {np.mean(mn < ml)*100:.1f}%  [69.7%]")
print(f"A Pearson r     : {np.corrcoef(P_lsqf[ok,0], P_nn[ok,0])[0,1]:.3f}  [0.996]")

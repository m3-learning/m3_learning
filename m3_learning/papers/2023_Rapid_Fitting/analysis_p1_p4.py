"""Priorities 1-4: SHO parameter agreement, reconstruction-MSE distribution + paired
test, unphysical-outlier counts, and train/test overfitting check.

Loads the cached LSQF SHO fit + the Adam epoch-4 checkpoint (NO retraining), runs
inference over all spectra, and prints copy-paste numeric summaries. Figures saved
to ./Figures/ as PNG (300 dpi) + SVG. Colors: LSQF #3B75AF, NN #EF8636.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn, SHO_Model
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

LSQF_C, NN_C = "#0072B2", "#E69F00"
LABELS = ["A", "omega_0", "Q", "phi"]
SHO_RANGES = [(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)]

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
args = ap.parse_args()

set_style("printing")
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP
set_pub_style()
random_seed(seed=42)

dataset = BE_Dataset(args.data, SHO_fit_func_LSQF=SHO_fit_func_nn)
dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
dataset = BE_Dataset(args.data, SHO_fit_func_LSQF=SHO_fit_func_nn)

# state matching the paper violin
state = {"fitter": "LSQF", "raw_format": "complex", "resampled": True,
         "scaled": True, "output_shape": "index", "measurement_state": "all"}
dataset.set_attributes(**state)

# build + load model (no retraining)
post = ComplexPostProcessor(dataset)
model_ = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4,
                            dataset.SHO_scaler, post)
model = Model(model_, dataset, training=True, model_basename="SHO_Fitter_original_data")
print(f"Loading checkpoint: {args.ckpt}")
model.load(args.ckpt)

X_data, _ = dataset.NN_data()
n = X_data.shape[0]
print(f"\nn spectra = {n}")

# ---- params ----
dataset.set_attributes(**state)
pred_data, _, nn_phys = model.predict(X_data)           # physical NN params
nn_phys = np.asarray(nn_phys)
dataset.scaled = False
lsqf_phys = dataset.SHO_fit_results().reshape(-1, 4)    # physical LSQF params
dataset.scaled = True
nn_scaled = dataset.SHO_scaler.transform(nn_phys)
lsqf_scaled = dataset.SHO_fit_results().reshape(-1, 4)  # scaled LSQF (violin 'true')

# =====================================================================
# PRIORITY 1a — parameter-space agreement (scaled space; corr/R2 are scale-free)
# =====================================================================
print("\n" + "=" * 78)
print("PRIORITY 1a  NN-vs-LSQF SHO parameter agreement  (n = %d, all spectra)" % n)
print("=" * 78)
hdr = f"{'param':>8} | {'medianAbsDiff':>13} {'IQR(diff)':>10} | {'Pearson':>8} {'Spearman':>9} {'R^2':>8}  (scaled)"
print(hdr); print("-" * len(hdr))
p1 = {}
for i, lab in enumerate(LABELS):
    a, b = lsqf_scaled[:, i], nn_scaled[:, i]
    diff = b - a
    mad = np.median(np.abs(diff))
    iqr = np.percentile(diff, 75) - np.percentile(diff, 25)
    pear = stats.pearsonr(a, b)[0]
    spear = stats.spearmanr(a, b)[0]
    r2 = r2_score(a, b)
    # physical-units median |diff| for context
    mad_phys = np.median(np.abs(nn_phys[:, i] - lsqf_phys[:, i]))
    p1[lab] = dict(mad=mad, iqr=iqr, pearson=pear, spearman=spear, r2=r2, mad_phys=mad_phys)
    print(f"{lab:>8} | {mad:13.4f} {iqr:10.4f} | {pear:8.4f} {spear:9.4f} {r2:8.4f}   (phys medianAbsDiff={mad_phys:.4g})")

# ---- P1b: 4-panel hexbin NN vs LSQF (physical) ----
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, (ax, lab) in enumerate(zip(axes, LABELS)):
    x, y = lsqf_phys[:, i], nn_phys[:, i]
    hb = ax.hexbin(x, y, gridsize=80, bins="log", cmap=SEQ_CMAP, mincnt=1)
    lo, hi = np.percentile(np.concatenate([x, y]), [0.1, 99.9])
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"LSQF  {lab}"); ax.set_ylabel(f"NN  {lab}")
    ax.set_title(f"{lab}   (Pearson r={p1[lab]['pearson']:.3f}, R²={p1[lab]['r2']:.3f})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="log10(count)")
fig.tight_layout()
fig.savefig("./Figures/Figure_param_agreement_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig("./Figures/Figure_param_agreement_scatter.svg", bbox_inches="tight")
plt.close(fig)
print("\nWrote ./Figures/Figure_param_agreement_scatter.png and .svg")

# =====================================================================
# PRIORITY 2 — per-spectrum reconstruction MSE + Wilcoxon paired test
# =====================================================================
def to_arr(lst):
    return np.rollaxis(np.array(lst), 0, 3)  # [N, freq, 2]

dataset.set_attributes(**state)
dataset.scaled = True
true_list, _ = dataset.raw_spectra(frequency=True)
nn_recon, _ = dataset.raw_spectra(fit_results=nn_phys, frequency=True)
lsqf_recon, _ = dataset.raw_spectra(fit_results=lsqf_phys, frequency=True)

t = to_arr(true_list)
mse_nn = Model.MSE(t, to_arr(nn_recon))
mse_lsqf = Model.MSE(t, to_arr(lsqf_recon))

def q(x):
    return np.mean(x), np.median(x), np.percentile(x, 25), np.percentile(x, 75)

print("\n" + "=" * 78)
print("PRIORITY 2a  Per-spectrum reconstruction MSE  (scaled real/imag space)")
print("=" * 78)
for name, m in [("LSQF", mse_lsqf), ("NN  ", mse_nn)]:
    mean, med, q1, q3 = q(m)
    print(f"  {name}: mean={mean:.5f}  median={med:.5f}  Q1={q1:.5f}  Q3={q3:.5f}")
W, p = stats.wilcoxon(mse_nn, mse_lsqf)               # paired, NN vs LSQF
med_paired = np.median(mse_nn - mse_lsqf)
frac_nn_better = np.mean(mse_nn < mse_lsqf)
print(f"\n  Wilcoxon signed-rank (NN vs LSQF): W={W:.4g}  p={p:.3e}")
print(f"  median paired diff (NN - LSQF) = {med_paired:+.6f}")
print(f"  fraction of spectra where NN beats LSQF = {frac_nn_better:.4f}")
verdict = ("BETTER (lower MSE)" if med_paired < 0 else "WORSE (higher MSE)") + \
          (" and the difference is statistically significant" if p < 0.05 else
           " but the difference is NOT statistically significant")
print(f"  -> The network is {verdict}.")

# ---- P2b: CDF + histogram ----
fig, (axc, axh) = plt.subplots(1, 2, figsize=(12, 4.2))
for m, c, lab in [(mse_lsqf, LSQF_C, "LSQF"), (mse_nn, NN_C, "NN")]:
    xs = np.sort(m); ys = np.arange(1, len(xs) + 1) / len(xs)
    axc.plot(xs, ys, color=c, lw=1.5, label=lab)
axc.set_xlabel("per-spectrum reconstruction MSE"); axc.set_ylabel("CDF")
axc.set_xlim(0, np.percentile(np.concatenate([mse_nn, mse_lsqf]), 99))
axc.legend(frameon=False); axc.set_title("CDF")
hi = np.percentile(np.concatenate([mse_nn, mse_lsqf]), 99)
bins = np.linspace(0, hi, 120)
axh.hist(mse_lsqf, bins=bins, color=LSQF_C, alpha=0.55, label="LSQF")
axh.hist(mse_nn, bins=bins, color=NN_C, alpha=0.55, label="NN")
axh.set_xlabel("per-spectrum reconstruction MSE"); axh.set_ylabel("count")
axh.legend(frameon=False); axh.set_title("Histogram")
fig.tight_layout()
fig.savefig("./Figures/Figure_mse_distribution.png", dpi=300, bbox_inches="tight")
fig.savefig("./Figures/Figure_mse_distribution.svg", bbox_inches="tight")
plt.close(fig)
print("Wrote ./Figures/Figure_mse_distribution.png and .svg")

# =====================================================================
# PRIORITY 3 — unphysical-outlier counts (physical params vs SHO_ranges)
# =====================================================================
print("\n" + "=" * 78)
print("PRIORITY 3  Out-of-physical-range counts  (criterion: value strictly outside")
print("            the codebase SHO_ranges; Q range is (-300,0) per the code's sign)")
print("=" * 78)
print(f"  ranges: A{SHO_RANGES[0]}  omega_0{SHO_RANGES[1]}  Q{SHO_RANGES[2]}  phi{SHO_RANGES[3]}")
print(f"  {'param':>8} | {'LSQF n_out':>11} {'LSQF %':>8} | {'NN n_out':>9} {'NN %':>8}")
print("  " + "-" * 52)
for i, lab in enumerate(LABELS):
    lo, hi = SHO_RANGES[i]
    l_out = int(np.sum((lsqf_phys[:, i] < lo) | (lsqf_phys[:, i] > hi)))
    n_out = int(np.sum((nn_phys[:, i] < lo) | (nn_phys[:, i] > hi)))
    print(f"  {lab:>8} | {l_out:11d} {100*l_out/n:8.3f} | {n_out:9d} {100*n_out/n:8.3f}")

# =====================================================================
# PRIORITY 4 — overfitting check (train vs test reconstruction MSE)
# =====================================================================
print("\n" + "=" * 78)
print("PRIORITY 4  Overfitting check (80/20 split, seed=42)")
print("=" * 78)
dataset.set_attributes(**state)
Xtr, Xte, _, _ = dataset.test_train_split_(test_size=0.2, random_state=42)
ptr, _, _ = model.predict(Xtr)
pte, _, _ = model.predict(Xte)
tr_mse = float(np.mean(Model.MSE(np.asarray(Xtr), np.asarray(ptr))))
te_mse = float(np.mean(Model.MSE(np.asarray(Xte), np.asarray(pte))))
print(f"  train MSE = {tr_mse:.6f}   (n={Xtr.shape[0]})")
print(f"  test  MSE = {te_mse:.6f}   (n={Xte.shape[0]})")
print(f"  test/train ratio = {te_mse/tr_mse:.4f}  -> "
      f"{'no meaningful overfitting' if te_mse/tr_mse < 1.05 else 'some gap, inspect'}")
print("\nDONE P1-P4")

"""E12: initialization-controlled least-squares baselines for the hysteresis loops.

Mirrors E3 (SHO) for the 9-parameter loop fits, per reviewer round 2:
 (a) refit every loop with unbounded LM least squares initialized from the
     NETWORK's predicted parameters;
 (b) refit every loop initialized from the cached LSQF loop parameters
     (tests whether the BGlib loop fit sits at a local optimum of the
     shared scaled objective);
 (c) multistart best-of-5 on a 2,000-loop subsample (NN init, LSQF init,
     3 multiplicative jitters of the LSQF init).

Objective identical to E7: MSE between the scaled loop reconstruction and the
scaled measured loop (GlobalScaler mean/std), 96 voltage steps per loop.
Validation anchors before any refit is trusted: cached-LSQF median 0.08983,
network median 0.00533 (paper checkpoint epoch 599).

Runs on CPU only. No timing claims.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

T0 = time.time()
OUT = "/tmp/e12"
PAPER_DIR = os.path.expanduser(
    "~/Desktop/Projects/m3_learning/m3_learning/papers/2023_Rapid_Fitting")
CKPT = os.path.join(
    OUT, "Hysteresis_Loop_Fitter_model_optimizer_Trust Region CG_epoch_599_"
         "train_loss_0.006038312106910679.pth")


def log(msg):
    print(f"[{time.time()-T0:8.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- data ----
os.chdir(PAPER_DIR)
from m3_learning.be.dataset import BE_Dataset            # noqa: E402
from m3_learning.be.nn import SHO_fit_func_nn            # noqa: E402
from m3_learning.be.loop_fitter import loop_fitting_function_torch  # noqa: E402
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model  # noqa: E402

log("loading dataset")
dataset = BE_Dataset("Data/data_raw.h5", SHO_fit_func_LSQF=SHO_fit_func_nn)

loops_lsqf, raw_scaled, voltage = dataset.get_LSQF_hysteresis_fits(compare=True)
scaler = dataset.hysteresis_scaler
MEAN, STD = float(scaler.mean), float(scaler.std)
raw_phys = raw_scaled * STD + MEAN                      # measured loops, physical
lsqf_params = np.asarray(dataset.LSQF_hysteresis_params()).reshape(-1, 9)
V = np.asarray(voltage[:, 0]).squeeze().astype(np.float64)
N, L = raw_scaled.shape
log(f"{N} loops x {L} steps; scaler mean {MEAN:.4g} std {STD:.4g}")

lsqf_scaled = scaler.transform(loops_lsqf)
mse_lsqf = ((lsqf_scaled - raw_scaled) ** 2).mean(axis=1)
log(f"anchor cached-LSQF median {np.median(mse_lsqf):.5f} (expect 0.08983)")

# -------------------------------------------------------------- network ----
log("loading network checkpoint + predicting")
import torch  # noqa: E402
data, _ = dataset.get_hysteresis(scaled=True, loop_interpolated=True)
model_ = Multiscale1DFitter(loop_fitting_function_torch, V, 1, 9,
                            dataset.loop_param_scaler,
                            loops_scaler=dataset.hysteresis_scaler)
model = Model(model_, dataset, training=False,
              path=OUT, model_basename="Hysteresis_Loop_Fitter")
model_.load_state_dict(torch.load(CKPT, map_location="cpu"))
# train-mode forward returns (scaled_fits, unscaled_params); the architecture
# has no dropout/batchnorm, so this is deterministic inference
model_.train()
X = torch.atleast_3d(torch.tensor(data.reshape(-1, L))).float()
chunks = []
with torch.no_grad():
    for i in range(0, len(X), 4096):
        _, up = model_(X[i:i + 4096])
        chunks.append(up.cpu().numpy())
nn_params = np.concatenate(chunks).reshape(-1, 9).astype(np.float64)
nn_loops = loop_fitting_function_torch(nn_params, V).cpu().detach().numpy().squeeze()
mse_nn = ((scaler.transform(nn_loops) - raw_scaled) ** 2).mean(axis=1)
log(f"anchor network median {np.median(mse_nn):.5f} (expect 0.00533)")

# ------------------------------------------------------------ loop model ----
H = L // 2
V1, V2 = V[:H], V[H:]


def loop_np(p, Vfull=None):
    a0, a1, a2, a3, a4, b0, b1, b2, b3 = p
    with np.errstate(all="ignore"):
        g1 = (b1 - b0) / 2 * (erf((V1 - a2) * 1000) + 1) + b0
        g2 = (b3 - b2) / 2 * (erf((V2 - a3) * 1000) + 1) + b2
        y1 = (g1 * erf((V1 - a2) / g1) + b0) / (b0 + b1)
        y2 = (g2 * erf((V2 - a3) / g2) + b2) / (b2 + b3)
        f1 = a0 + a1 * y1 + a4 * V1
        f2 = a0 + a1 * y2 + a4 * V2
    return np.concatenate([f1, f2])


# self-check the numpy port against the torch implementation
chk = loop_fitting_function_torch(lsqf_params[:64], V).cpu().detach().numpy().squeeze()
chk_np = np.stack([loop_np(p) for p in lsqf_params[:64]])
port_dev = float(np.nanmax(np.abs(chk - chk_np)))
log(f"numpy port max deviation vs torch: {port_dev:.3e}")
assert port_dev < 1e-8


def residual(p, target_phys):
    r = (loop_np(p) - target_phys) / STD
    return np.where(np.isfinite(r), r, 1e6)


def refit(p0, target_phys):
    try:
        res = least_squares(residual, p0, args=(target_phys,), method="lm",
                            max_nfev=4000)
        return res.x, float(np.mean(residual(res.x, target_phys) ** 2))
    except Exception:
        return p0, float(np.mean(residual(p0, target_phys) ** 2))


def run_block(name, inits):
    out = np.empty(N)
    t = time.time()
    for i in range(N):
        _, out[i] = refit(inits[i], raw_phys[i])
        if i % 2000 == 1999:
            log(f"  {name}: {i+1}/{N} ({(time.time()-t)/(i+1)*1000:.1f} ms/loop)")
    return out


log("E12(a): refit from NETWORK inits, all loops")
mse_refit_nn = run_block("nn-init", nn_params)
np.save(os.path.join(OUT, "mse_refit_nn.npy"), mse_refit_nn)
log("E12(b): refit from LSQF inits, all loops")
mse_refit_lsqf = run_block("lsqf-init", lsqf_params)
np.save(os.path.join(OUT, "mse_refit_lsqf.npy"), mse_refit_lsqf)
np.save(os.path.join(OUT, "mse_nn.npy"), mse_nn)
np.save(os.path.join(OUT, "mse_lsqf_cached.npy"), mse_lsqf)

log("E12(c): multistart best-of-5, 2000-loop subsample")
rng = np.random.default_rng(42)
sub = rng.choice(N, 2000, replace=False)
mse_multi = np.empty(len(sub))
for k, i in enumerate(sub):
    starts = [nn_params[i], lsqf_params[i]]
    for j in range(3):
        starts.append(lsqf_params[i] * rng.normal(1.0, 0.2, 9))
    mse_multi[k] = min(refit(s, raw_phys[i])[1] for s in starts)
    if k % 500 == 499:
        log(f"  multistart: {k+1}/{len(sub)}")

# ------------------------------------------------------------- summary ----
def stats(x):
    return dict(median=float(np.median(x)), mean=float(np.mean(x)),
                q1=float(np.quantile(x, 0.25)), q3=float(np.quantile(x, 0.75)))

def boot_median_diff(a, b, n=10000):
    # pixel-clustered bootstrap: 4 loops per pixel, loop index is pixel-major
    rng2 = np.random.default_rng(42)
    d4 = (a - b).reshape(-1, 4)
    diffs = np.empty(n)
    for t in range(n):
        sel = rng2.integers(0, d4.shape[0], d4.shape[0])
        diffs[t] = np.median(d4[sel].ravel())
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]

result = {
    "anchors": {"lsqf_cached_median": float(np.median(mse_lsqf)),
                "nn_median": float(np.median(mse_nn)),
                "numpy_port_max_dev": port_dev},
    "cached_LSQF": stats(mse_lsqf),
    "network": stats(mse_nn),
    "refit_nn_init": stats(mse_refit_nn),
    "refit_lsqf_init": stats(mse_refit_lsqf),
    "multistart_sub2000": stats(mse_multi),
    "network_sub2000": stats(mse_nn[sub]),
    "fractions": {
        "refit_nn_beats_network": float((mse_refit_nn < mse_nn).mean()),
        "refit_lsqf_beats_cached": float((mse_refit_lsqf < mse_lsqf).mean()),
        "refit_lsqf_beats_network": float((mse_refit_lsqf < mse_nn).mean()),
        "network_beats_refit_lsqf": float((mse_nn < mse_refit_lsqf).mean()),
        "multistart_beats_network_sub": float((mse_multi < mse_nn[sub]).mean()),
    },
    "pixel_clustered_CI_network_minus_refit_lsqf":
        boot_median_diff(mse_nn, mse_refit_lsqf),
    "notes": "objective = scaled-space MSE identical to E7; unbounded LM "
             "(loop fitter has no configured bounds); numpy loop function "
             "verified against torch to <1e-8; multistart = NN init + LSQF "
             "init + 3 multiplicative N(1,0.2) jitters of LSQF init, seed 42.",
}
with open(os.path.join(OUT, "e12_loop_refit.json"), "w") as f:
    json.dump(result, f, indent=1)
log("SUMMARY " + json.dumps({k: v for k, v in result.items()
                             if k in ("cached_LSQF", "network", "refit_nn_init",
                                      "refit_lsqf_init", "multistart_sub2000",
                                      "fractions")}, indent=1))
log("DONE")

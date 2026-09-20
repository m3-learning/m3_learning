"""E13: clean re-run of the ground-truth recovery study with fair baselines.

Improvements over the phase-3 (E1) protocol, per external review:
  - family (a) truth sampled WITHOUT replacement (no duplicate leak across split)
  - multistart bounded LSQF (heuristic + 4 jitters, best-of-5) as the refined baseline
  - empirical-prior MAP estimator (GMM prior fit on TRAIN-split truth only)
  - NN retrained with seeds 42/43/44 at (a, n=2) and (a, n=8) for seed variability
Same physics/recipe otherwise (generator validated against E1 to <1%: sigma_a
3.894e-3 vs 3.900e-3, in-range fraction 0.9630 exact). E1 (A10) remains an
independent replication; all E13 numbers are self-consistent on one set.

Stages checkpoint to /tmp/e13/work so a crash never loses compute.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

T0 = time.time()
WORK = "/tmp/e13/work"
os.makedirs(WORK, exist_ok=True)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, os.path.expanduser("~/Desktop/Projects/m3_learning/m3_learning/src"))

H5 = os.path.expanduser(
    "~/Desktop/Projects/m3_learning/m3_learning/papers/2023_Rapid_Fitting/Data/data_raw.h5")
LB = np.array([0.0, 1.31e6, -300.0, -np.pi])
UB = np.array([1.5e-4, 1.33e6, 0.0, np.pi])
NOISES = [0, 2, 4, 6, 8]


def log(m):
    print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)


def sho_np(p, w):
    A, w0, Q, phi = (np.asarray(p)[:, i][:, None] for i in range(4))
    return A * np.exp(1j * phi) * w0**2 / (w**2 - 1j * w * w0 / Q - w0**2)


# ============================ STAGE A: data =================================
def stage_a():
    import h5py
    f = h5py.File(H5, "r")
    fit = f["Measurement_000/Channel_000/Raw_Data-SHO_Fit_000/Fit"][()]
    P = np.stack([fit["Amplitude [V]"], fit["Frequency [Hz]"],
                  fit["Quality Factor"], fit["Phase [rad]"]], -1
                 ).reshape(-1, 4).astype(np.float64)
    sv = f["Measurement_000/Channel_000/Spectroscopic_Values"][()]
    w = np.sort(np.unique(sv[0]).astype(np.float64))
    assert len(w) == 165
    finite = np.isfinite(P).all(axis=1)
    inr = finite & (P > LB).all(axis=1) & (P < UB).all(axis=1)
    pool = P[inr]
    log(f"pool {pool.shape}, in-range {inr.mean():.4f}")

    # family a: 200k WITHOUT replacement
    rng = np.random.default_rng(42)
    a_truth = pool[rng.permutation(len(pool))[:200000]]

    # family b: 100k best clean fits by scaled recon MSE
    raw = f["Measurement_000/Channel_000/Raw_Data"][()].reshape(-1, 165).astype(np.complex128)
    mu_re, sd_re = raw.real.mean(), raw.real.std()
    mu_im, sd_im = raw.imag.mean(), raw.imag.std()
    rec = np.empty_like(raw)
    for i in range(0, len(P), 100000):
        rec[i:i+100000] = sho_np(P[i:i+100000], w)
    mse = (((rec.real - raw.real) / sd_re)**2).mean(1) + (((rec.imag - raw.imag) / sd_im)**2).mean(1)
    mse[~finite] = np.inf
    b_truth = P[np.argsort(mse)[:100000]]
    del raw, rec

    data = {"w": w}
    from sklearn.model_selection import train_test_split
    for fam, truth in (("a", a_truth), ("b", b_truth)):
        clean = sho_np(truth, w)
        sigma = float(np.std(clean))
        idx = np.arange(len(truth))
        tr, te = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
        scal = {"mu_re": float(clean.real.mean()), "sd_re": float(clean.real.std()),
                "mu_im": float(clean.imag.mean()), "sd_im": float(clean.imag.std())}
        np.savez(f"{WORK}/data_{fam}.npz", truth=truth, clean=clean.astype(np.complex64),
                 tr=tr, te=te, sigma=sigma, **scal)
        log(f"family {fam}: N={len(truth)} sigma={sigma:.6e} "
            f"(E1: {'3.900e-3' if fam=='a' else '3.237e-3'})")
        data[fam] = True
    np.save(f"{WORK}/w.npy", w)
    log("STAGE A done")


def load_fam(fam):
    d = np.load(f"{WORK}/data_{fam}.npz")
    return d


def noisy(clean, fam, n):
    if n == 0:
        return np.array(clean, dtype=np.complex128)
    d = load_fam(fam)
    sigma = float(d["sigma"])
    rng = np.random.default_rng(10000 + (0 if fam == "a" else 5000) + n)
    re = clean.real + rng.uniform(-n * sigma, n * sigma, clean.shape)
    im = clean.imag + rng.uniform(-n * sigma, n * sigma, clean.shape)
    return re + 1j * im


# ============================ STAGE B: NN ===================================
def stage_b():
    import torch
    torch.set_num_threads(8)
    from m3_learning.be.nn import SHO_fit_func_nn
    from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor
    from m3_learning.nn.random import random_seed
    w = np.load(f"{WORK}/w.npy")

    class LatentScaler:
        pass

    class ChannelScaler:
        def __init__(self, mean, std):
            self.mean, self.std = mean, std

    class RawScaler:
        def __init__(self, d):
            self.real_scaler = ChannelScaler(float(d["mu_re"]), float(d["sd_re"]))
            self.imag_scaler = ChannelScaler(float(d["mu_im"]), float(d["sd_im"]))

    class MockDataset:
        pass

    jobs = [(f, n, 42) for f in ("a", "b") for n in NOISES]
    jobs += [("a", 2, 43), ("a", 2, 44), ("a", 8, 43), ("a", 8, 44)]

    results = {}
    for fam, n, seed in jobs:
        tag = f"{fam}_n{n}_s{seed}"
        outp = f"{WORK}/nn_{tag}.npz"
        if os.path.exists(outp):
            log(f"NN {tag}: cached"); continue
        d = load_fam(fam)
        truth, tr, te = d["truth"], d["tr"], d["te"]
        X = noisy(d["clean"].astype(np.complex128), fam, n)
        rs = RawScaler(d)
        Xs = np.stack([(X.real - rs.real_scaler.mean) / rs.real_scaler.std,
                       (X.imag - rs.imag_scaler.mean) / rs.imag_scaler.std], -1).astype(np.float32)
        ls = LatentScaler()
        ls.mean_ = truth[tr].mean(0); ls.var_ = truth[tr].var(0)
        ls.mean_[3] = 0.0; ls.var_[3] = 1.0          # phase unscaled
        ds = MockDataset(); ds.raw_data_scaler = rs; ds.SHO_scaler = ls
        ds.frequency_bin = w; ds.noise = n
        post = ComplexPostProcessor(ds)
        random_seed(seed=seed)
        model_ = Multiscale1DFitter(SHO_fit_func_nn, w, 2, 4, ls, post)
        model = Model(model_, ds, training=True, path=WORK, model_basename=f"E13_{tag}")
        t = time.time()
        model.fit(Xs[tr], batch_size=500, epochs=5, seed=seed, optimizer="Adam")
        log(f"NN {tag}: trained in {time.time()-t:.0f}s")
        model_.train()
        preds = []
        with torch.no_grad():
            XT = torch.tensor(Xs[te])
            for i in range(0, len(XT), 4096):
                _, up = model_(XT[i:i+4096])
                preds.append(up.cpu().numpy())
        np.savez(outp, params=np.concatenate(preds))
        log(f"NN {tag}: predicted {len(te)} test spectra")
    log("STAGE B done")


# ====================== STAGE C: LSQF heuristic/multistart ==================
_FITCTX = {}


def _init_fit(w, scal, sigma):
    from BGlib.be.analysis.utils.be_sho import SHOestimateGuess
    _FITCTX.update(w=w, scal=scal, sigma=sigma, guess=SHOestimateGuess)


def _residual(p, target, w, scal):
    r = sho_np(p[None], w)[0]
    rr = (r.real - target.real) / scal[1]
    ri = (r.imag - target.imag) / scal[3]
    out = np.concatenate([rr, ri])
    return np.where(np.isfinite(out), out, 1e6)


def _fit_one(args):
    from scipy.optimize import least_squares
    spec, mode, jseed = args
    w, scal = _FITCTX["w"], _FITCTX["scal"]

    def clipwrap(p):
        p = np.array(p, dtype=np.float64)
        if p[2] > 0:            # guess collapses to +Q under heavy noise;
            p[2] = -p[2]        # canonicalize to this pipeline's -Q branch
        p[3] = np.mod(p[3] + np.pi, 2 * np.pi) - np.pi
        return np.clip(p, LB + 1e-12, UB - 1e-12)

    try:
        g = clipwrap(_FITCTX["guess"](spec, w))
    except Exception:
        g = clipwrap((LB + UB) / 2)
    starts = [g]
    if mode == "multistart":
        rng = np.random.default_rng(jseed)
        for _ in range(4):
            starts.append(clipwrap(g * rng.normal(1.0, 0.2, 4)))
    best, bcost = None, np.inf
    for s in starts:
        try:
            res = least_squares(_residual, s, args=(spec, w, scal),
                                bounds=(LB, UB), method="trf", max_nfev=1000)
            if res.cost < bcost:
                best, bcost = res.x, res.cost
        except Exception:
            pass
    return best if best is not None else g


def stage_c():
    w = np.load(f"{WORK}/w.npy")
    for fam in ("a", "b"):
        d = load_fam(fam)
        te = d["te"]
        scal = (float(d["mu_re"]), float(d["sd_re"]), float(d["mu_im"]), float(d["sd_im"]))
        for n in NOISES:
            for mode in ("heuristic", "multistart"):
                outp = f"{WORK}/lsqf_{mode}_{fam}_n{n}.npy"
                if os.path.exists(outp):
                    continue
                X = noisy(d["clean"].astype(np.complex128), fam, n)[te]
                args = [(X[i], mode, 777 + i) for i in range(len(X))]
                t = time.time()
                with Pool(7, initializer=_init_fit, initargs=(w, scal, float(d["sigma"]))) as p:
                    out = p.map(_fit_one, args, chunksize=200)
                np.save(outp, np.array(out))
                log(f"LSQF {mode} {fam} n={n}: {len(X)} fits in {time.time()-t:.0f}s")
    log("STAGE C done")


# ============================ STAGE D: MAP ==================================
_MAPCTX = {}


def _init_map(w, scal, gmm_params, nvar):
    from BGlib.be.analysis.utils.be_sho import SHOestimateGuess
    _MAPCTX.update(w=w, scal=scal, gmm=gmm_params, nvar=nvar, guess=SHOestimateGuess)


def _gmm_neglogpdf(theta_std, gmm):
    wts, means, precs, logdets = gmm
    x = theta_std
    comps = []
    for k in range(len(wts)):
        dx = x - means[k]
        comps.append(np.log(wts[k]) - 0.5 * (dx @ precs[k] @ dx) + 0.5 * logdets[k]
                     - 2 * np.log(2 * np.pi))
    m = max(comps)
    return -(m + np.log(sum(np.exp(c - m) for c in comps)))


def _map_one(args):
    from scipy.optimize import minimize
    spec, = args
    w, scal, gmm, nvar = _MAPCTX["w"], _MAPCTX["scal"], _MAPCTX["gmm"], _MAPCTX["nvar"]
    mu = gmm[4]; sd = gmm[5]

    def nll(p):
        r = _residual(p, spec, w, scal)
        data_term = 0.5 * np.sum(r**2) * (scal[1]**2 / nvar)   # unscale then /nvar
        return data_term + _gmm_neglogpdf((p - mu) / sd, gmm[:4])

    def clipwrap(p):
        p = np.array(p, dtype=np.float64)
        if p[2] > 0:
            p[2] = -p[2]
        p[3] = np.mod(p[3] + np.pi, 2 * np.pi) - np.pi
        return np.clip(p, LB + 1e-9, UB - 1e-9)

    try:
        starts = [clipwrap(_MAPCTX["guess"](spec, w))]
    except Exception:
        starts = [clipwrap((LB + UB) / 2)]
    for k in range(2):
        starts.append(clipwrap(mu + sd * gmm[1][k]))
    best, bval = starts[0], np.inf
    for s in starts:
        try:
            res = minimize(nll, s, method="L-BFGS-B",
                           bounds=list(zip(LB + 1e-9, UB - 1e-9)),
                           options={"maxiter": 200})
            if res.fun < bval:
                best, bval = res.x, res.fun
        except Exception:
            pass
    return best


def stage_d():
    from sklearn.mixture import GaussianMixture
    w = np.load(f"{WORK}/w.npy")
    for fam in ("a", "b"):
        d = load_fam(fam)
        truth, tr, te = d["truth"], d["tr"], d["te"]
        scal = (float(d["mu_re"]), float(d["sd_re"]), float(d["mu_im"]), float(d["sd_im"]))
        mu, sd = truth[tr].mean(0), truth[tr].std(0)
        Z = (truth[tr] - mu) / sd
        gm = GaussianMixture(4, covariance_type="full", random_state=0).fit(Z[::10])
        precs = [np.linalg.inv(c) for c in gm.covariances_]
        logdets = [float(np.linalg.slogdet(pc)[1]) for pc in precs]
        gmm = (gm.weights_, gm.means_, precs, logdets, mu, sd)
        rng = np.random.default_rng(4242)
        sub = rng.choice(len(te), 10000, replace=False)
        for n in NOISES[1:]:
            outp = f"{WORK}/map_{fam}_n{n}.npy"
            if os.path.exists(outp):
                continue
            sigma = float(d["sigma"])
            nvar = (n * sigma)**2 / 3.0
            X = noisy(d["clean"].astype(np.complex128), fam, n)[te][sub]
            t = time.time()
            with Pool(7, initializer=_init_map, initargs=(w, scal, gmm, nvar)) as p:
                out = p.map(_map_one, [(x,) for x in X], chunksize=100)
            np.save(outp, np.array(out))
            np.save(f"{WORK}/map_sub_{fam}.npy", sub)
            log(f"MAP {fam} n={n}: {len(X)} in {time.time()-t:.0f}s")
    log("STAGE D done")


# ============================ STAGE E: metrics ==============================
def errs(pred, truth):
    wrap = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
    # canonicalize the exact (A, phi) -> (-A, phi+pi) degeneracy of the SHO
    # response onto the A>0 branch used by the truth and the bounded fitters
    pred = np.array(pred, dtype=np.float64)
    neg = pred[:, 0] < 0
    pred[neg, 0] *= -1
    pred[neg, 3] = wrap(pred[neg, 3] + np.pi)
    return {
        "A": float(np.median(np.abs(pred[:, 0] - truth[:, 0]))),
        "w0": float(np.median(np.abs(pred[:, 1] - truth[:, 1]))),
        "Q": float(np.median(np.abs(pred[:, 2] - truth[:, 2]))),
        "phi": float(np.median(np.abs(wrap(pred[:, 3] - truth[:, 3])))),
    }


def stage_e():
    out = {}
    for fam in ("a", "b"):
        d = load_fam(fam)
        truth_te = d["truth"][d["te"]]
        sub = np.load(f"{WORK}/map_sub_{fam}.npy") if os.path.exists(f"{WORK}/map_sub_{fam}.npy") else None
        for n in NOISES:
            e = {"sigma": float(d["sigma"])}
            for mode in ("heuristic", "multistart"):
                p = f"{WORK}/lsqf_{mode}_{fam}_n{n}.npy"
                if os.path.exists(p):
                    e[f"lsqf_{mode}"] = errs(np.load(p), truth_te)
            nnp = f"{WORK}/nn_{fam}_n{n}_s42.npz"
            if os.path.exists(nnp):
                e["nn_s42"] = errs(np.load(nnp)["params"], truth_te)
            for s in (43, 44):
                sp = f"{WORK}/nn_{fam}_n{n}_s{s}.npz"
                if os.path.exists(sp):
                    e[f"nn_s{s}"] = errs(np.load(sp)["params"], truth_te)
            mp = f"{WORK}/map_{fam}_n{n}.npy"
            if os.path.exists(mp) and sub is not None:
                e["map"] = errs(np.load(mp), truth_te[sub])
                e["nn_s42_on_map_sub"] = errs(np.load(nnp)["params"][sub], truth_te[sub]) if os.path.exists(nnp) else None
                e["lsqf_multistart_on_map_sub"] = errs(np.load(f"{WORK}/lsqf_multistart_{fam}_n{n}.npy")[sub], truth_te[sub]) if os.path.exists(f"{WORK}/lsqf_multistart_{fam}_n{n}.npy") else None
            out[f"{fam}_n{n}"] = e
    with open("/tmp/e13/e13_results.json", "w") as fjson:
        json.dump(out, fjson, indent=1)
    log("STAGE E done -> /tmp/e13/e13_results.json")


if __name__ == "__main__":
    stages = sys.argv[1] if len(sys.argv) > 1 else "abcde"
    if "a" in stages and not os.path.exists(f"{WORK}/data_b.npz"):
        stage_a()
    if "b" in stages:
        stage_b()
    if "c" in stages:
        stage_c()
    if "d" in stages:
        stage_d()
    if "e" in stages:
        stage_e()
    log("ALL DONE")

"""Clean-layout SHO switching-maps grids (Figure_5_47 noise-7 LSQF/Adam/TR-CG, and
Figure_15 noise-0 Adam single method). Full content kept: 9 steps x 4 params x N
methods. Built from scratch with explicit axes geometry so row labels, column
headers, step numbers and per-parameter colorbars never collide. cividis, square
tiles, landscape SI canvas."""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, SEQ_CMAP

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn, find_best_model
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

DATA = "Data/data_raw.h5"
ADAM0 = ("Trained Models/SHO Fitter/"
         "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.03403592322973526.pth")
NAMES = ["A", "ω₀", "Q", "φ"]
CLIMS = [(0, 1.4e-4), (1.31e6, 1.33e6), (-230, -160), (-np.pi, np.pi)]
NSTEP = 9

set_style("printing"); random_seed(seed=42); set_pub_style()
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
basepath = sorted(glob.glob("Trained Models/SHO Fitter/*_nn_benchmarks_noise"))[-1]
results = find_best_model(basepath, "Batch_Trainging_SpeedTest.csv")
SIDE = int(round(np.sqrt(dataset.num_pix)))


def load_model(noise, opt):
    ck = basepath + "/" + results[(noise, opt)]['filename'].split("//")[-1]
    fitter = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4,
                                dataset.SHO_scaler, ComplexPostProcessor(dataset))
    m = Model(fitter, dataset, training=False, model_basename="SHO_Fitter_original_data")
    m.load(ck)
    return m


def get_params(noise, models):
    """returns list of physical param arrays [N,4] for LSQF + each model."""
    st = {"resampled": True, "raw_format": "complex", "fitter": "LSQF", "scaled": False,
          "output_shape": "index", "measurement_state": "all", "resampled_bins": 165,
          "LSQF_phase_shift": np.pi / 2, "NN_phase_shift": None, "noise": noise}
    dataset.set_attributes(**st)
    out = [np.asarray(dataset.SHO_fit_results(state=st)).reshape(-1, 4)]
    X, _ = dataset.NN_data()
    for m in models:
        out.append(np.asarray(dataset.SHO_fit_results(model=m, phase_shift=np.pi / 2, X_data=X)).reshape(-1, 4))
    return out


def to_maps(params):
    """physical [N,4] -> (num_pix, cycle_steps, 4) off-state cycle-2 maps."""
    dataset.measurement_state = "off"; dataset.cycle = 2
    a = np.asarray(params).reshape(dataset.num_pix, dataset.voltage_steps, 4)
    a = dataset.get_measurement_cycle(a, cycle=2, axis=1)
    return np.asarray(a)


def cycle_voltage():
    dataset.measurement_state = "off"; dataset.cycle = 2
    v = dataset.dc_voltage
    v = dataset.get_cycle(v)
    return np.asarray(v).squeeze()


def plot_grid(method_arrays, labels, filename):
    nm = len(method_arrays)
    steps = method_arrays[0].shape[1]
    inds = np.linspace(0, steps - 1, NSTEP, dtype=int)
    volt = cycle_voltage()
    vinds = np.linspace(0, len(volt) - 1, NSTEP, dtype=int)

    # geometry (inches)
    tile, gut, blkgap, rgap = 0.46, 1.25, 0.34, 0.08
    topv_h, v_gap, hdr_h = 1.0, 0.55, 0.34
    stepnum_h, cbar_gap, cbar_h = 0.3, 0.32, 0.2
    top_m, rmargin = 0.12, 0.35
    block_w = NSTEP * tile
    maps_w = 4 * block_w + 3 * blkgap
    bands_h = nm * tile + (nm - 1) * rgap
    fig_w = gut + maps_w + rmargin
    fig_h = top_m + topv_h + v_gap + hdr_h + bands_h + stepnum_h + cbar_gap + cbar_h + 0.35

    fig = plt.figure(figsize=(fig_w, fig_h))

    def axbox(left, top, w, h):
        return fig.add_axes([left / fig_w, (fig_h - top - h) / fig_h, w / fig_w, h / fig_h])

    def block_left(p):
        return gut + p * (block_w + blkgap)

    # voltage plot (spans the maps width)
    axv = axbox(gut, top_m, maps_w, topv_h)
    axv.plot(volt, "k", lw=1.4)
    for n, vi in enumerate(vinds):
        axv.plot(vi, volt[vi], "o", color="k", ms=6)
        axv.annotate(str(n + 1), (vi, volt[vi]), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=8)
    axv.set_ylabel("Voltage (V)"); axv.set_xlabel("Step"); axv.margins(x=0.01)

    top_hdr = top_m + topv_h + v_gap
    top_bands = top_hdr + hdr_h
    # param column headers
    for p in range(4):
        fig.text((block_left(p) + block_w / 2) / fig_w,
                 (fig_h - (top_hdr + hdr_h / 2)) / fig_h, NAMES[p],
                 ha="center", va="center", fontsize=14)
    # method row labels (once per band, horizontal, in the gutter)
    for k in range(nm):
        yc = top_bands + k * (tile + rgap) + tile / 2
        fig.text((gut - 0.12) / fig_w, (fig_h - yc) / fig_h, labels[k],
                 ha="right", va="center", fontsize=12)
    # tiles
    for k in range(nm):
        top = top_bands + k * (tile + rgap)
        for p in range(4):
            for sidx, ind in enumerate(inds):
                ax = axbox(block_left(p) + sidx * tile, top, tile * 0.94, tile * 0.94)
                m2d = method_arrays[k][:, ind, p].reshape(SIDE, SIDE)
                ax.imshow(m2d, cmap=SEQ_CMAP, vmin=CLIMS[p][0], vmax=CLIMS[p][1],
                          aspect="auto", origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
    # step numbers under bottom band
    top_steps = top_bands + bands_h + 0.02
    for p in range(4):
        for sidx in range(NSTEP):
            fig.text((block_left(p) + sidx * tile + tile / 2) / fig_w,
                     (fig_h - (top_steps + stepnum_h / 2)) / fig_h, str(sidx + 1),
                     ha="center", va="center", fontsize=8)
    # one horizontal colorbar per parameter
    top_cbar = top_bands + bands_h + stepnum_h + cbar_gap
    for p in range(4):
        cax = axbox(block_left(p) + block_w * 0.08, top_cbar, block_w * 0.84, cbar_h)
        sm = ScalarMappable(norm=Normalize(*CLIMS[p]), cmap=SEQ_CMAP)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", format="%.2g")
        cb.set_label(NAMES[p], fontsize=11); cax.tick_params(labelsize=8)

    fig.savefig(f"./Figures/{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"./Figures/{filename}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote ./Figures/{filename}.png and .svg  (fig {fig_w:.1f}x{fig_h:.1f} in)", flush=True)


# ---- Figure_5_47: noise 7, LSQF + Adam + TR-CG (per-noise best) ----
ma = load_model(7, "Adam"); mt = load_model(7, "Trust Region CG")
p7 = get_params(7, [ma, mt])
maps7 = [to_maps(p) for p in p7]
plot_grid(maps7, ["LSQF", "Adam (network)", "TR-CG (network)"],
          "Figure_5_47_switching_maps_comparison_7_noise")

# ---- Figure_15: noise 0, single Adam model ----
m0 = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4,
                        dataset.SHO_scaler, ComplexPostProcessor(dataset))
mdl0 = Model(m0, dataset, training=False, model_basename="SHO_Fitter_original_data"); mdl0.load(ADAM0)
st0 = {"resampled": True, "raw_format": "complex", "fitter": "LSQF", "scaled": False,
       "output_shape": "index", "measurement_state": "all", "resampled_bins": 165,
       "LSQF_phase_shift": np.pi / 2, "NN_phase_shift": None, "noise": 0}
dataset.set_attributes(**st0)
X0, _ = dataset.NN_data()
nn0 = np.asarray(dataset.SHO_fit_results(model=mdl0, phase_shift=np.pi / 2, X_data=X0)).reshape(-1, 4)
plot_grid([to_maps(nn0)], ["network (Adam)"], "Figure_15_NN_Switching_Maps")
print("DONE switchmaps")

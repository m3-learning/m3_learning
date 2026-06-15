"""Regenerate Figure_14_LSQF_NN_bmw_comparison (SHO best/median/worst, nb2 recipe)
with the corrected shared legend. Loads the Adam epoch-4 checkpoint, no retraining."""
import numpy as np
from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.be.viz import Viz
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

DATA = "Data/data_raw.h5"
ADAM = ("Trained Models/SHO Fitter/"
        "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.03403592322973526.pth")
RANGES = [(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)]

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

post = ComplexPostProcessor(dataset)
m_ = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4, dataset.SHO_scaler, post)
model = Model(m_, dataset, training=True, model_basename="SHO_Fitter_original_data")
model.load(ADAM)

dataset.NN_phase_shift = np.pi / 2
dataset.LSQF_phase_shift = np.pi / 2
dataset.measurement_state = "all"

true_state = {"fitter": "LSQF", "raw_format": "complex", "resampled": True,
              "scaled": True, "output_shape": "index", "measurement_state": "all"}
out_state = {"scaled": True, "raw_format": "magnitude spectrum"}

LSQF = BE_viz.get_best_median_worst(true_state, prediction={"fitter": "LSQF"},
                                    out_state=out_state, SHO_results=True, n=1)
NN = BE_viz.get_best_median_worst(true_state, prediction=model,
                                  out_state=out_state, SHO_results=True, n=1)

BE_viz.SHO_Fit_comparison(
    (LSQF, NN), ["LSQF", "NN"],
    model_comparison=[model, {"fitter": "LSQF"}],
    out_state=out_state,
    size=(1.7, 1.7), gaps=(1.9, 1.9),   # roomier panels so the bigger pub-style labels don't overlap
    filename="Figure_14_LSQF_NN_bmw_comparison",
)
print("Wrote ./Figures/Figure_14_LSQF_NN_bmw_comparison.png and .svg")
print("DONE fig14")

"""Regenerate hysteresis_comparison (nb5 recipe) with the corrected shared legend.
Loads the hysteresis Trust-Region-CG checkpoint, no retraining."""
import numpy as np
import torch
from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.be.viz import Viz
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.be.loop_fitter import loop_fitting_function_torch
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model

DATA = "Data/data_raw.h5"
HYST = ("Trained Models/Hysteresis Fitter/"
        "Hysteresis_Loop_Fitter_model_optimizer_Trust Region CG_epoch_599_train_loss_0.006081502470705245.pth")

set_style("printing"); random_seed(seed=42)
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP
set_pub_style()
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
printing = printer(basepath="./Figures/")
BE_viz = Viz(dataset, printing, verbose=False, image_scalebar=[2000, 500, "nm", "br"])

# hysteresis data + scalers (cached LSQF loop fit in the h5)
data, voltage = dataset.get_hysteresis(scaled=True, loop_interpolated=True)
model_ = Multiscale1DFitter(loop_fitting_function_torch, voltage[:, 0].squeeze(),
                            1, 9, dataset.loop_param_scaler,
                            loops_scaler=dataset.hysteresis_scaler)
model = Model(model_, dataset, path='Trained Models/Hysteresis Fitter/',
              training=True, model_basename="Hysteresis_Loop_Fitter")
model.load(HYST)

BE_viz.hysteresis_comparison(['LSQF', 'NN'],
                             row=None, col=None, cycle=None,
                             size=(1.25, 1.25), gaps=(0.9, 0.9),
                             nn_model=model, measurement_state=None,
                             filename="hysteresis_comparison")
print("Wrote ./Figures/hysteresis_comparison.png and .svg")
print("DONE hyst_comp")

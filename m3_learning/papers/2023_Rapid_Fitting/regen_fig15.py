"""Regenerate Figure_15_NN_Switching_Maps (single-model NN parameter maps, nb2/nb6
recipe) with cividis + pub style. Adam epoch-4 model on noise-0 data, no retraining."""
import numpy as np
from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.be.viz import Viz
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from pubstyle import set_pub_style, METHOD_COLORS, SEQ_CMAP

DATA = "Data/data_raw.h5"
ADAM = ("Trained Models/SHO Fitter/"
        "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.03403592322973526.pth")
RANGES = [(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)]

set_style("printing"); random_seed(seed=42); set_pub_style()
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
printing = printer(basepath="./Figures/")
BE_viz = Viz(dataset, printing, verbose=False, SHO_ranges=RANGES,
             image_scalebar=[2000, 500, "nm", "br"])

dataset.set_attributes(fitter="LSQF", raw_format="complex", resampled=True,
                       scaled=True, output_shape="index", measurement_state="all", noise=0)
post = ComplexPostProcessor(dataset)
m_ = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4, dataset.SHO_scaler, post)
model = Model(m_, dataset, training=False, model_basename="SHO_Fitter_original_data")
model.load(ADAM)

X_data, _ = dataset.NN_data()
_, _, parm = model.predict(X_data)
BE_viz.SHO_switching_maps(np.asarray(parm), filename="Figure_15_NN_Switching_Maps")
print("Wrote ./Figures/Figure_15_NN_Switching_Maps.png and .svg")
print("DONE fig15")

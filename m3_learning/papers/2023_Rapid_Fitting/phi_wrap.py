"""Circular (wrapped) phi agreement — the honest manuscript numbers for phi, since the
raw linear stats are dominated by 2pi wrapping / branch convention."""
import numpy as np
from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter, Model, ComplexPostProcessor

DATA = "Data/data_raw.h5"
CKPT = ("Trained Models/SHO Fitter/"
        "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4_train_loss_0.03403592322973526.pth")
set_style("printing"); random_seed(seed=42)
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
dataset = BE_Dataset(DATA, SHO_fit_func_LSQF=SHO_fit_func_nn)
st = {"fitter": "LSQF", "raw_format": "complex", "resampled": True, "scaled": True,
      "output_shape": "index", "measurement_state": "all"}
dataset.set_attributes(**st)
post = ComplexPostProcessor(dataset)
m_ = Multiscale1DFitter(SHO_fit_func_nn, dataset.frequency_bin, 2, 4, dataset.SHO_scaler, post)
model = Model(m_, dataset, training=True, model_basename="SHO_Fitter_original_data")
model.load(CKPT)
_, _, nn = model.predict(dataset.NN_data()[0]); nn = np.asarray(nn)
dataset.scaled = False
lsqf = dataset.SHO_fit_results().reshape(-1, 4)
nphi, lphi = nn[:, 3], lsqf[:, 3]
# wrap the difference to (-pi, pi]
dwrap = np.angle(np.exp(1j * (nphi - lphi)))
# align NN phase onto the LSQF branch, then check genuinely-out-of-range
nphi_wrapped = np.angle(np.exp(1j * nphi))
n = len(nphi)
print(f"n = {n}")
print(f"phi  median|wrapped diff| = {np.median(np.abs(dwrap)):.4f} rad")
print(f"phi  IQR(wrapped diff)    = {np.percentile(dwrap,75)-np.percentile(dwrap,25):.4f} rad")
print(f"phi  circular agreement: frac |wrapped diff| < 0.5 rad = {np.mean(np.abs(dwrap)<0.5):.4f}")
print(f"phi  NN out of (-pi,pi) RAW     = {np.mean((nphi< -np.pi)|(nphi>np.pi))*100:.3f}%")
print(f"phi  NN out of (-pi,pi) WRAPPED = {np.mean((nphi_wrapped< -np.pi)|(nphi_wrapped>np.pi))*100:.3f}%")

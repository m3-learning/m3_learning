"""Regenerate the publication-clean SHO violin (Figure_16_Violin) from cached data.

Run this on a machine that has the experiment h5 (with the cached LSQF SHO fit)
and the trained Adam checkpoint -- e.g. the Lambda box / persistent NFS that ran
the full pipeline. It loads the saved weights (no retraining), runs inference,
and renders the violin with the fixed publication styling in be/viz.py.

Usage (from papers/2023_Rapid_Fitting/, with the SDK installed):
    python regenerate_violins.py [--data PATH_TO_h5] [--ckpt PATH_TO_pth]

Defaults assume the standard layout: ./Data/data_raw.h5 and the Adam checkpoint
under ./Trained Models/SHO Fitter/. Output: ./Figures/Figure_16_Violin.png/.svg
"""
import argparse
import glob
import numpy as np

from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.be.viz import Viz
from m3_learning.be.dataset import BE_Dataset
from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.nn.Fitter1D.Fitter1D import (
    Multiscale1DFitter,
    Model,
    ComplexPostProcessor,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./Data/data_raw.h5",
                    help="path to the experiment h5 with the cached LSQF SHO fit")
    ap.add_argument("--ckpt", default=None,
                    help="path to the trained Adam .pth checkpoint "
                         "(default: auto-find under ./Trained Models/SHO Fitter/)")
    args = ap.parse_args()

    set_style("printing")
    random_seed(seed=42)

    printing = printer(basepath="./Figures/")

    # dataset + cached LSQF fit (SHO_Fitter is idempotent -- instant if already fit)
    dataset = BE_Dataset(args.data, SHO_fit_func_LSQF=SHO_fit_func_nn)
    dataset.SHO_Fitter(force=False, h5_sho_targ_grp="Raw_Data_SHO_Fit")
    dataset = BE_Dataset(args.data, SHO_fit_func_LSQF=SHO_fit_func_nn)

    BE_viz = Viz(
        dataset, printing, verbose=True,
        SHO_ranges=[(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)],
        image_scalebar=[2000, 500, "nm", "br"],
    )

    # rebuild the network and LOAD the trained Adam weights (no retraining)
    postprocessor = ComplexPostProcessor(dataset)
    model_ = Multiscale1DFitter(
        SHO_fit_func_nn, dataset.frequency_bin, 2, 4,
        dataset.SHO_scaler, postprocessor,
    )
    model = Model(model_, dataset, training=True,
                  model_basename="SHO_Fitter_original_data")

    ckpt = args.ckpt
    if ckpt is None:
        hits = sorted(glob.glob(
            "Trained Models/SHO Fitter/"
            "SHO_Fitter_original_data_model_optimizer_Adam_epoch_4*.pth"))
        if not hits:
            raise SystemExit(
                "No Adam checkpoint found under 'Trained Models/SHO Fitter/'. "
                "Pass one with --ckpt.")
        ckpt = hits[0]
    print(f"Loading checkpoint: {ckpt}")
    model.load(ckpt)

    X_data, _ = dataset.NN_data()

    true_state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": True,
        "scaled": True,
        "output_shape": "index",
        "measurement_state": "all",
    }
    BE_viz.violin_plot_comparison(true_state, model, X_data,
                                  filename="Figure_16_Violin")
    print("Wrote ./Figures/Figure_16_Violin.png and .svg")


if __name__ == "__main__":
    main()

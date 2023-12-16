from m3_learning.be.nn import SHO_fit_func_nn, SHO_Model, static_state_decorator, clear_all_tensors
from m3_learning.util.system_info import SystemInfo
import argparse
from datetime import datetime
import torch
import os
import numpy as np
from m3_learning.optimizers.TrustRegion import TRCG
from m3_learning.nn.random import random_seed
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.be.viz import Viz
from m3_learning.be.dataset import BE_Dataset
import itertools
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Specify the filename and the path to save the file
filename = "data_raw_unmod.h5"
save_path = "/home/ferroelectric/Documents/m3_learning/m3_learning/papers/2023_Rapid_Fitting/Data/"
optimizer_TR = {"name": "TRCG", "optimizer": TRCG,
                "radius": 5, "device": "cuda", "ADAM_epochs": 2}


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Batch Training')

    # Add arguments
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--optimizers', required=True, nargs='+', help='List of optimizers')
    parser.add_argument('--noise_list', required=True, nargs='+', help='List of noise values')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--seed', type=int, required=True, help='Seed value')
    parser.add_argument('--write_CSV', default='Batch_Training_SpeedTest.csv', help='CSV file name')
    parser.add_argument('--basepath', default='', help='Base path for saving files')
    parser.add_argument('--early_stopping_time', type=int, default=None, help='Early stopping time')

    # Parse arguments
    args = parser.parse_args()

    # Call the function
    batch_training(
        dataset=args.dataset,
        optimizers=args.optimizers,
        noise_list=args.noise_list,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        write_CSV=args.write_CSV,
        basepath=args.basepath,
        early_stopping_time=args.early_stopping_time
    )

if __name__ == '__main__':
    main()
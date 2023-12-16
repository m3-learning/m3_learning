from m3_learning.be.nn import SHO_fit_func_nn, batch_training
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


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Specify the filename and the path to save the file
filename = "data_raw_unmod.h5"
save_path = "/home/ferroelectric/Documents/m3_learning/m3_learning/papers/2023_Rapid_Fitting/Data/"
optimizer_TR = {"name": "TRCG", "optimizer": TRCG,
                "radius": 5, "device": "cuda", "ADAM_epochs": 2}
optimizers = ['Adam', optimizer_TR]
noise_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
batch_size = [500, 1000, 5000, 10000]
epochs = [5]
seed = [0, 41, 43, 44, 45, 46]
early_stopping_time = 60*4
basepath_postfix = 'nn_benchmarks_noise'

# Original filename
csv_name = 'nn_benchmarks_noise.csv'

print(f'using torch version {torch.__version__}')

printing = printer(basepath='./Figures/')

set_style("printing")
random_seed(seed=42)

data_path = save_path + "/" + filename


# instantiate the dataset object
dataset = BE_Dataset(data_path, SHO_fit_func_LSQF=SHO_fit_func_nn)

# print the contents of the file
dataset.print_be_tree()

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time in a 'pretty' format (e.g., YYYY-MM-DD_HH-MM-SS)
formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')

basepath = f'{formatted_datetime}_{basepath_postfix}'

# Usage example
system_info = SystemInfo()
cpu_info, gpu_info = system_info.get_system_info()
system_info.save_to_file("Trained Models/" + basepath,
                         "system_info.txt", cpu_info, gpu_info)

# Generate all combinations
combinations = list(itertools.product(
    optimizers, noise_list, batch_size, epochs, seed))

for i, training in enumerate(combinations):

    optimizer_ = training[0]
    noise_ = training[1]
    seed_ = training[4]

    print(i, optimizer_, noise_, seed_)

batch_training(dataset, optimizers, noise_list, batch_size, epochs,
               seed,
               write_CSV="Batch_Trainging_SpeedTest.csv",
               basepath=basepath,
               early_stopping_time=early_stopping_time,
               skip=432
               )

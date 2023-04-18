"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np
import pyUSID as usid
import h5py
import torch
import time
from scipy import special
import tensorflow as tf
import os
from BGlib import be as belib
import sidpy
import numpy as np
from pyUSID.io.hdf_utils import  reshape_to_n_dims, get_auxiliary_datasets
from sidpy.hdf.hdf_utils import get_attr


def transform_params(params_real, params_pred):
    """Utility function to transform the parameters to the correct distribution

    Args:
        params_real (np.array): real parameters
        params_pred (np.array): predicted parameters

    Returns:
        tuple(np.array, np.array): returns the tuple of updated real and predicted parameters
    """
    params_pred[:, 3] = params_pred[:, 3] + 1/2 * np.pi
    params_real[:, 3] = params_real[:, 3] + 1/2 * np.pi

    params_pred[:, 3] = np.where(
        params_pred[:, 3] < np.pi, params_pred[:, 3], params_pred[:, 3] - 2 * np.pi)
    params_real[:, 3] = np.where(
        params_real[:, 3] < np.pi, params_real[:, 3], params_real[:, 3] - 2 * np.pi)

    params_pred[:, 3] = np.where(
        params_pred[:, 2] > 0, params_pred[:, 3], params_pred[:, 3] + np.pi)
    params_real[:, 3] = np.where(
        params_real[:, 2] > 0, params_real[:, 3], params_real[:, 3] + np.pi)

    params_pred[:, 2] = np.where(
        params_pred[:, 2] > 0, params_pred[:, 2], params_pred[:, 2]*-1)
    params_real[:, 2] = np.where(
        params_real[:, 2] > 0, params_real[:, 2], params_real[:, 2]*-1)

    params_pred[:, 3] = np.where(
        params_pred[:, 3] < np.pi, params_pred[:, 3], params_pred[:, 3] - 2 * np.pi)
    params_real[:, 3] = np.where(
        params_real[:, 3] < np.pi, params_real[:, 3], params_real[:, 3] - 2 * np.pi)
    
    params_pred[:, 0] = np.abs(params_pred[:, 0])

    return params_real, params_pred


def convert_amp_phase(data):
  """Utility function to extract the magnitude and phase from complex data

  Args:
      data (np.complex): raw complex data from BE spectroscopies

  Returns:
      np.array: returns the magnitude and the phase
  """
  magnitude = np.abs(data)
  phase = np.angle(data)
  return magnitude, phase

def SHO_Fitter(input_file_path, force = False, max_cores = -1, max_mem=1024*8):
    """Function that computes the SHO fit results

    Args:
        input_file_path (str): path to the h5 file
        force (bool, optional): forces the SHO results to be computed from scratch. Defaults to False.
        max_cores (int, optional): number of processor cores to use. Defaults to -1.
        max_mem (_type_, optional): maximum ram to use. Defaults to 1024*8.
    """
    
    start_time_lsqf = time.time()

    (data_dir, filename) = os.path.split(input_file_path)

    if input_file_path.endswith(".h5"):
        # No translation here
        h5_path = input_file_path

        tl = belib.translators.LabViewH5Patcher()
        tl.translate(h5_path, force_patch=force)

    else:
        pass

    folder_path, h5_raw_file_name = os.path.split(h5_path)
    h5_file = h5py.File(h5_path, "r+")
    print("Working on:\n" + h5_path)

    h5_main = usid.hdf_utils.find_dataset(h5_file, "Raw_Data")[0]

    h5_pos_inds = h5_main.h5_pos_inds
    pos_dims = h5_main.pos_dim_sizes
    pos_labels = h5_main.pos_dim_labels
    print(pos_labels, pos_dims)

    h5_meas_grp = h5_main.parent.parent

    parm_dict = sidpy.hdf_utils.get_attributes(h5_meas_grp)

    expt_type = usid.hdf_utils.get_attr(h5_file, "data_type")

    is_ckpfm = expt_type == "cKPFMData"
    if is_ckpfm:
        num_write_steps = parm_dict["VS_num_DC_write_steps"]
        num_read_steps = parm_dict["VS_num_read_steps"]
        num_fields = 2

    if expt_type != "BELineData":
        vs_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_mode")
        try:
            field_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_measure_in_field_loops")
        except KeyError:
            print("field mode could not be found. Setting to default value")
            field_mode = "out-of-field"
        try:
            vs_cycle_frac = usid.hdf_utils.get_attr(h5_meas_grp, "VS_cycle_fraction")
        except KeyError:
            print("VS cycle fraction could not be found. Setting to default value")
            vs_cycle_frac = "full"

    sho_fit_points = 5  # The number of data points at each step to use when fitting
    sho_override = force  # Force recompute if True

    

    h5_sho_targ_grp = None
    h5_sho_file_path = os.path.join(
        folder_path, h5_raw_file_name)
    
    
    print("\n\nSHO Fits will be written to:\n" + h5_sho_file_path + "\n\n")
    f_open_mode = "w"
    if os.path.exists(h5_sho_file_path):
        f_open_mode = "r+"
    h5_sho_file = h5py.File(h5_sho_file_path, mode=f_open_mode)
    h5_sho_targ_grp = h5_sho_file

    sho_fitter = belib.analysis.BESHOfitter(
        h5_main, cores=max_cores, verbose=False, h5_target_group=h5_sho_targ_grp
    )
    sho_fitter.set_up_guess(
        guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian,
        num_points=sho_fit_points,
    )
    h5_sho_guess = sho_fitter.do_guess(override=sho_override)
    sho_fitter.set_up_fit()
    h5_sho_fit = sho_fitter.do_fit(override=sho_override)
    parms_dict = parms_dict = sidpy.hdf_utils.get_attributes(h5_main.parent.parent)

    print(f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters")
    
def SHO_fit_to_array(fit_results, channels = 5):
    """Utility function to convert the SHO fit results to an array

    Args:
        fit_results (h5 Dataset): Location of the fit results in an h5 file

    Returns:
        np.array: SHO fit results
    """
    # create a list for parameters
    fit_results_list = []
    for sublist in np.array(
        fit_results
    ):
        for item in sublist:
            for i in item:
                fit_results_list.append(i)

    # flatten parameters list into numpy array
    fit_results_list = np.array(fit_results_list).reshape(fit_results.shape[0], fit_results.shape[1], channels)

    return fit_results_list

def fit_loop_function(h5_file, h5_sho_fit, loop_success = False, h5_loop_group = None,\
                      results_to_new_file = False, max_mem=1024*8, max_cores = None):
    """_summary_

    Args:
        h5_file (_type_): _description_
        h5_sho_fit (_type_): _description_
        loop_success (bool, optional): _description_. Defaults to False.
        h5_loop_group (_type_, optional): _description_. Defaults to None.
        max_mem (_type_, optional): _description_. Defaults to 1024*8.
        max_cores (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    
    expt_type = sidpy.hdf.hdf_utils.get_attr(h5_file, 'data_type')
    h5_meas_grp = h5_sho_fit.parent.parent.parent
    vs_mode = sidpy.hdf.hdf_utils.get_attr(h5_meas_grp, 'VS_mode')
    try:
        vs_cycle_frac = sidpy.hdf.hdf_utils.get_attr(h5_meas_grp, 'VS_cycle_fraction')
    except KeyError:
        print('VS cycle fraction could not be found. Setting to default value')
        vs_cycle_frac = 'full'
    if results_to_new_file:
        h5_loop_file_path = os.path.join(folder_path, 
                                         h5_raw_file_name.replace('.h5', '_loop_fit.h5'))
        print('\n\nLoop Fits will be written to:\n' + h5_loop_file_path + '\n\n')
        f_open_mode = 'w'
        if os.path.exists(h5_loop_file_path):
            f_open_mode = 'r+'
        h5_loop_file = h5py.File(h5_loop_file_path, mode=f_open_mode)
        h5_loop_group = h5_loop_file
    loop_fitter = belib.analysis.BELoopFitter(h5_sho_fit, expt_type, vs_mode, vs_cycle_frac,
                                           cores=max_cores, h5_target_group=h5_loop_group, 
                                           verbose=False)
    loop_fitter.set_up_guess()
    h5_loop_guess = loop_fitter.do_guess(override=False)
    
    # Calling explicitly here since Fitter won't do it automatically
    h5_guess_loop_parms = loop_fitter.extract_loop_parameters(h5_loop_guess)
    loop_fitter.set_up_fit()
    h5_loop_fit = loop_fitter.do_fit(override=False)
    h5_loop_group = h5_loop_fit.parent
    loop_success = True
    return h5_loop_fit, h5_loop_group

def loop_lsqf(h5_f):
    step_chan='DC_Offset'
    cmap=None

    h5_projected_loops = h5_f['Measurement_000']['Channel_000']['Raw_Data-SHO_Fit_000']['Guess-Loop_Fit_000']['Projected_Loops']
    h5_loop_guess = h5_f['Measurement_000']['Channel_000']['Raw_Data-SHO_Fit_000']['Guess-Loop_Fit_000']['Guess']
    h5_loop_fit = h5_f['Measurement_000']['Channel_000']['Raw_Data-SHO_Fit_000']['Guess-Loop_Fit_000']['Fit']

    # Prepare some variables for plotting loops fits and guesses
    # Plot the Loop Guess and Fit Results
    proj_nd, _ = reshape_to_n_dims(h5_projected_loops)
    guess_nd, _ = reshape_to_n_dims(h5_loop_guess)
    fit_nd, _ = reshape_to_n_dims(h5_loop_fit)

    h5_projected_loops = h5_loop_guess.parent['Projected_Loops']
    h5_proj_spec_inds = get_auxiliary_datasets(h5_projected_loops,
                                            aux_dset_name='Spectroscopic_Indices')[-1]
    h5_proj_spec_vals = get_auxiliary_datasets(h5_projected_loops,
                                            aux_dset_name='Spectroscopic_Values')[-1]
    h5_pos_inds = get_auxiliary_datasets(h5_projected_loops,
                                        aux_dset_name='Position_Indices')[-1]
    pos_nd, _ = reshape_to_n_dims(h5_pos_inds, h5_pos=h5_pos_inds)
    pos_dims = list(pos_nd.shape[:h5_pos_inds.shape[1]])
    pos_labels = get_attr(h5_pos_inds, 'labels')


    # reshape the vdc_vec into DC_step by Loop
    spec_nd, _ = reshape_to_n_dims(h5_proj_spec_vals, h5_spec=h5_proj_spec_inds)
    loop_spec_dims = np.array(spec_nd.shape[1:])
    loop_spec_labels = get_attr(h5_proj_spec_vals, 'labels')

    spec_step_dim_ind = np.where(loop_spec_labels == step_chan)[0][0]

    # # move the step dimension to be the first after all position dimensions
    rest_loop_dim_order = list(range(len(pos_dims), len(proj_nd.shape)))
    rest_loop_dim_order.pop(spec_step_dim_ind)
    new_order = list(range(len(pos_dims))) + [len(pos_dims) + spec_step_dim_ind] + rest_loop_dim_order

    new_spec_order = np.array(new_order[len(pos_dims):], dtype=np.uint32) - len(pos_dims)

    # Also reshape the projected loops to Positions-DC_Step-Loop
    final_loop_shape = pos_dims + [loop_spec_dims[spec_step_dim_ind]] + [-1]
    proj_nd2 = np.moveaxis(proj_nd, spec_step_dim_ind + len(pos_dims), len(pos_dims))
    proj_nd_3 = np.reshape(proj_nd2, final_loop_shape)

    # Do the same for the guess and fit datasets
    guess_3d = np.reshape(guess_nd, pos_dims + [-1])
    fit_3d = np.reshape(fit_nd, pos_dims + [-1])

    # Get the bias vector:
    spec_nd2 = np.moveaxis(spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)
    bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims):])

    # Shift the bias vector and the loops by a quarter cycle
    shift_ind = int(-1 * bias_vec.shape[0] / 4)
    bias_shifted = np.roll(bias_vec, shift_ind, axis=0)
    proj_nd_shifted = np.roll(proj_nd_3, shift_ind, axis=len(pos_dims))

    return proj_nd_shifted


def loop_fitting_function(type, V, y):

    if(type == '9 parameters'):
        a0 = y[:, 0]
        a1 = y[:, 1]
        a2 = y[:, 2]
        a3 = y[:, 3]
        a4 = y[:, 4]
        b0 = y[:, 5]
        b1 = y[:, 6]
        b2 = y[:, 7]
        b3 = y[:, 8]
        d = 1000
        V1 = V[:int(len(V) / 2)]
        V2 = V[int(len(V) / 2):]

        g1 = (b1 - b0) / 2 * (special.erf((V1 - a2) * d) + 1) + b0
        g2 = (b3 - b2) / 2 * (special.erf((V2 - a3) * d) + 1) + b2

        y1 = (g1 * special.erf((V1 - a2) / g1) + b0) / (b0 + b1)
        y2 = (g2 * special.erf((V2 - a3) / g2) + b2) / (b2 + b3)

        f1 = a0 + a1 * y1 + a4 * V1
        f2 = a0 + a1 * y2 + a4 * V2

        loop_eval = np.vstack((f1, f2))
        return loop_eval
    
    elif(type == '13 parameters'):
        a1 = y[:, 0]
        a2 = y[:, 1]
        a3 = y[:, 2]
        b1 = y[:, 3]
        b2 = y[:, 4]
        b3 = y[:, 5]
        b4 = y[:, 6]
        b5 = y[:, 7]
        b6 = y[:, 8]
        b7 = y[:, 9]
        b8 = y[:, 10]
        Au = y[:, 11]
        Al = y[:, 12]

        # See supporting information for more information about the form of this function
        S1 = ((b1 + b2) / 2) + ((b2 - b1) / 2) * special.erf((V - b7) / b5)
        S2 = ((b4 + b3) / 2) + ((b3 - b4) / 2) * special.erf((V - b8) / b6)
        Branch1 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            special.erf((V - Au) / S1) + a3 * V
        Branch2 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            special.erf((V - Al) / S2) + a3 * V

        return np.concatenate((Branch1, np.flipud(Branch2)), axis=0).squeeze()
    else:
        print('No such parameters')
        return None
    
    
def loop_fitting_function_torch(type, V, y):

    if(type == '9 parameters'):
        a0 = y[:, 0].type(torch.float64)
        a1 = y[:, 1].type(torch.float64)
        a2 = y[:, 2].type(torch.float64)
        a3 = y[:, 3].type(torch.float64)
        a4 = y[:, 4].type(torch.float64)
        b0 = y[:, 5].type(torch.float64)
        b1 = y[:, 6].type(torch.float64)
        b2 = y[:, 7].type(torch.float64)
        b3 = y[:, 8].type(torch.float64)
        d = 1000
        V1 = torch.tensor(V[:int(len(V) / 2)]).cuda()
        V2 = torch.tensor(V[int(len(V) / 2):]).cuda()

        g1 = (b1 - b0) / 2 * (torch.erf((V1 - a2) * d) + 1) + b0
        g2 = (b3 - b2) / 2 * (torch.erf((V2 - a3) * d) + 1) + b2

        y1 = (g1 * torch.erf((V1 - a2) / g1) + b0) / (b0 + b1)
        y2 = (g2 * torch.erf((V2 - a3) / g2) + b2) / (b2 + b3)

        f1 = a0 + a1 * y1 + a4 * V1
        f2 = a0 + a1 * y2 + a4 * V2

        loop_eval = torch.transpose(torch.cat((f1, f2), axis=0), 1, 0)
        return loop_eval
    elif(type == '13 parameters'):
        a1 = y[:, 0].type(torch.float64)
        a2 = y[:, 1].type(torch.float64)
        a3 = y[:, 2].type(torch.float64)
        b1 = y[:, 3].type(torch.float64)
        b2 = y[:, 4].type(torch.float64)
        b3 = y[:, 5].type(torch.float64)
        b4 = y[:, 6].type(torch.float64)
        b5 = y[:, 7].type(torch.float64)
        b6 = y[:, 8].type(torch.float64)
        b7 = y[:, 9].type(torch.float64)
        b8 = y[:, 10].type(torch.float64)
        Au = y[:, 11].type(torch.float64)
        Al = y[:, 12].type(torch.float64)

        # See supporting information for more information about the form of this function
        S1 = ((b1 + b2) / 2) + ((b2 - b1) / 2) * torch.erf((V - b7) / b5)
        S2 = ((b4 + b3) / 2) + ((b3 - b4) / 2) * torch.erf((V - b8) / b6)
        Branch1 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            torch.erf((V - Au) / S1) + a3 * V
        Branch2 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            torch.erf((V - Al) / S2) + a3 * V

        return torch.squeeze(torch.cat((Branch1, torch.flipud(Branch2)), axis=0))
    else:
        print('No such parameters')
        return None
    

def loop_fitting_function_tf(type, V, y):
    if(type == '9 parameters'):
        a0 = y[:, 0]
        a1 = y[:, 1]
        a2 = y[:, 2]
        a3 = y[:, 3]
        a4 = y[:, 4]
        b0 = y[:, 5]
        b1 = y[:, 6]
        b2 = y[:, 7]
        b3 = y[:, 8]
        d = 1000
        V1 = V[:int(len(V) / 2)]
        V2 = V[int(len(V) / 2):]

        g1 = tf.add(tf.multiply(tf.divide(tf.subtract(b1, b0), 2), tf.add(tf.math.erf(tf.multiply(tf.subtract(V1, a2), d)), 1)), b0)
        g2 = tf.add(tf.multiply(tf.divide(tf.subtract(b3, b2), 2), tf.add(tf.math.erf(tf.multiply(tf.subtract(V2, a3), d)), 1)), b2)

        y1 = tf.divide(tf.add(tf.multiply(g1, tf.math.erf(tf.divide(tf.subtract(V1, a2), g1))), b0), tf.add(b0, b1))
        y2 = tf.divide(tf.add(tf.multiply(g2, tf.math.erf(tf.divide(tf.subtract(V2, a3), g2))), b2), tf.add(b2, b3))

        f1 = tf.add(a0, tf.add(tf.multiply(a1, y1), tf.multiply(a4, V1)))
        f2 = tf.add(a0, tf.add(tf.multiply(a1, y2), tf.multiply(a4, V2)))

        return tf.transpose(tf.concat([f1, f2], axis=0))

    elif(type == '13 parameters'):

        a1 = y[:, 0]
        a2 = y[:, 1]
        a3 = y[:, 2]
        b1 = y[:, 3]
        b2 = y[:, 4]
        b3 = y[:, 5]
        b4 = y[:, 6]
        b5 = y[:, 7]
        b6 = y[:, 8]
        b7 = y[:, 9]
        b8 = y[:, 10]
        Au = y[:, 11]
        Al = y[:, 12]

        epsilon = 5e-14

        S1 = tf.add(tf.divide(tf.add(b1, b2) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(b2,
                                                                                                b1) + epsilon, 2.0), tf.math.erf(tf.divide(tf.subtract(V, b7) + epsilon, b5))))
        S2 = tf.add(tf.divide(tf.add(b4, b3) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(b3,
                                                                                                b4) + epsilon, 2.0), tf.math.erf(tf.divide(tf.subtract(np.flipud(V), b8) + epsilon, b6))))
        Branch1 = tf.add(tf.add(tf.divide(tf.add(a1, a2) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(
            a2, a1) + epsilon, 2.0), tf.math.erf(tf.divide(tf.subtract(V, Au) + epsilon, S1)))), tf.multiply(a3, V))
        Branch2 = tf.add(tf.add(tf.divide(tf.add(a1, a2) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(a2, a1) + epsilon,
                                                                                                2.0), tf.math.erf(tf.divide(tf.subtract(np.flipud(V), Al) + epsilon, S2)))), tf.multiply(a3, np.flipud(V)))

        return tf.transpose(tf.concat([Branch1, Branch2], axis=0))
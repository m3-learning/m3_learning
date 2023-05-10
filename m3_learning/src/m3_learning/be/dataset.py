from m3_learning.util.h5_util import print_tree
from BGlib import be as belib
import pyUSID as usid
import os
import sidpy
import numpy as np
import h5py
import time
from m3_learning.util.h5_util import make_dataset, make_group, find_groups_with_string
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from m3_learning.viz.layout import layout_fig
# from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy import fftpack
from sklearn.preprocessing import StandardScaler
from m3_learning.util.preprocessing import global_scaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from m3_learning.be.processing import convert_amp_phase
from sklearn.model_selection import train_test_split
from m3_learning.be.nn import SHO_fit_func_nn

import m3_learning



def resample(y, num_points, axis=0):
   # Get the shape of the input array
    shape = y.shape

    # Swap the selected axis with the first axis
    y = np.swapaxes(y, axis, 0)

    # Create a new array of x values that covers the range of the original x values with the desired number of points
    x = np.arange(shape[axis])
    new_x = np.linspace(x.min(), x.max(), num_points)

    # Use cubic spline interpolation to estimate the y values of the curve at the new x values
    f = interp1d(x, y, kind='linear', axis=0)
    new_y = f(new_x)

    # Swap the first axis back with the selected axis
    new_y = np.swapaxes(new_y, axis, 0)

    return new_y


class BE_Dataset:

    def __init__(self, file_,
                 scaled=False,
                 raw_format="complex",
                 fitter='LSQF',
                 output_shape='pixels',
                 measurement_state='all',
                 resampled=False,
                 resampled_bins=None,
                 LSQF_phase_shift=None,
                 NN_phase_shift=None,
                 verbose=False,
                 noise = 0,
                 SHO_fit_func_LSQF=SHO_fit_func_nn,
                 **kwargs):

        self.file = file_
        self.resampled = resampled
        self.scaled = scaled
        self.raw_format = raw_format
        self.fitter = fitter
        self.output_shape = output_shape
        self.measurement_state = measurement_state
        
        # if None assigns it to the length of the original data
        if resampled_bins is None:
            resampled_bins = self.num_bins
            
        self.resampled_bins = resampled_bins
        
        self.LSQF_phase_shift = LSQF_phase_shift
        self.NN_phase_shift = NN_phase_shift
        self.verbose = verbose
        self.SHO_fit_func_LSQF = SHO_fit_func_LSQF
        self.resampled_data = {}
        self.noise = noise 

        for key, value in kwargs.items():
            setattr(self, key, value)

        # make only run if SHO exist
        self.set_preprocessing()

    def set_preprocessing(self):
        # extract the raw data and reshapes is
        self.set_raw_data()
        
        # resamples the data if necessary
        self.set_raw_data_resampler()
        
        # computes the scalar on the raw data
        self.raw_data_scaler = self.Raw_Data_Scaler(self.raw_data())
        
        
        try:
            self.set_SHO_LSQF()
        except:
            pass

    def default_state(self):
        default_state_ = {'raw_format': "complex",
                          "fitter": 'LSQF',
                          "output_shape": "pixels",
                          "scaled": False,
                          "measurement_state": "all",
                          "resampled": False,
                          "resampled_bins": 80,
                          "LSQF_phase_shift": None,
                          "NN_phase_shift": None, }

        self.set_state(**default_state_)

    def print_be_tree(self):
        """Utility file to print the Tree of a BE Dataset

        Args:
            path (str): path to the h5 file
        """

        with h5py.File(self.file, "r+") as h5_f:

            # Inspects the h5 file
            usid.hdf_utils.print_tree(h5_f)

            # prints the structure and content of the file
            print(
                "Datasets and datagroups within the file:\n------------------------------------")
            print_tree(h5_f.file)

            print("\nThe main dataset:\n------------------------------------")
            print(h5_f)
            print("\nThe ancillary datasets:\n------------------------------------")
            print(h5_f.file["/Measurement_000/Channel_000/Position_Indices"])
            print(h5_f.file["/Measurement_000/Channel_000/Position_Values"])
            print(
                h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Indices"])
            print(
                h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Values"])

            print(
                "\nMetadata or attributes in a datagroup\n------------------------------------")
            for key in h5_f.file["/Measurement_000"].attrs:
                print("{} : {}".format(
                    key, h5_f.file["/Measurement_000"].attrs[key]))

    def data_writer(self, base, name, data):
        with h5py.File(self.file, "r+") as h5_f:
            try:
                make_dataset(h5_f[base],
                             name,
                             data)
            except:
                self.delete(f"{base}/{name}")
                make_dataset(h5_f[base],
                             name,
                             data)

    # delete a dataset
    def delete(self, name):
        with h5py.File(self.file, "r+") as h5_f:
            try:
                del h5_f[name]
            except KeyError:
                print("Dataset not found, could not be deleted")

    def SHO_Fitter(self, force=False, max_cores=-1, max_mem=1024*8, 
                   dataset_name = "Raw_Data",
                   h5_sho_targ_grp = None):
        """Function that computes the SHO fit results

        Args:
            force (bool, optional): forces the SHO results to be computed from scratch. Defaults to False.
            max_cores (int, optional): number of processor cores to use. Defaults to -1.
            max_mem (_type_, optional): maximum ram to use. Defaults to 1024*8.
        """
        with h5py.File(self.file, "r+") as h5_file:
            # TODO fix delete
            # if force:
            #     self.delete(None)

            # the start time of the fit
            start_time_lsqf = time.time()

            # splits the directory path and the file name
            (data_dir, filename) = os.path.split(self.file)


            if self.file.endswith(".h5"):
                # No translation here
                h5_path = self.file

                # tl = belib.translators.LabViewH5Patcher()
                # tl.translate(h5_path, force_patch=force)

            else:
                pass
            
            # splits the path and the folder name
            folder_path, h5_raw_file_name = os.path.split(h5_path)
            
            # h5_file = h5py.File(h5_path, "r+")
            print("Working on:\n" + h5_path)

            # get the main dataset
            h5_main = usid.hdf_utils.find_dataset(h5_file, dataset_name)[0]

            # grabs some useful parameters from the dataset
            h5_pos_inds = h5_main.h5_pos_inds
            pos_dims = h5_main.pos_dim_sizes
            pos_labels = h5_main.pos_dim_labels
            print(pos_labels, pos_dims)

            # gets the measurement group name
            h5_meas_grp = h5_main.parent.parent

            # gets all of the attributes of the grout
            parm_dict = sidpy.hdf_utils.get_attributes(h5_meas_grp)

            # gets the data type of the dataset
            expt_type = usid.hdf_utils.get_attr(h5_file, "data_type")
            

            # code for using cKPFM Data
            is_ckpfm = expt_type == "cKPFMData"
            if is_ckpfm:
                num_write_steps = parm_dict["VS_num_DC_write_steps"]
                num_read_steps = parm_dict["VS_num_read_steps"]
                num_fields = 2

            if expt_type != "BELineData":
                vs_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_mode")
                try:
                    field_mode = usid.hdf_utils.get_attr(
                        h5_meas_grp, "VS_measure_in_field_loops")
                except KeyError:
                    print("field mode could not be found. Setting to default value")
                    field_mode = "out-of-field"
                try:
                    vs_cycle_frac = usid.hdf_utils.get_attr(
                        h5_meas_grp, "VS_cycle_fraction")
                except KeyError:
                    print("VS cycle fraction could not be found. Setting to default value")
                    vs_cycle_frac = "full"

            sho_fit_points = 5  # The number of data points at each step to use when fitting
            sho_override = force  # Force recompute if True

            # h5_sho_targ_grp = None
            h5_sho_file_path = os.path.join(
                folder_path, h5_raw_file_name)

            print("\n\nSHO Fits will be written to:\n" + h5_sho_file_path + "\n\n")
            f_open_mode = "w"
            if os.path.exists(h5_sho_file_path):
                f_open_mode = "r+"
            h5_sho_file = h5py.File(h5_sho_file_path, mode=f_open_mode)
            
            if h5_sho_targ_grp is None:
                h5_sho_targ_grp = h5_sho_file
            else: 
                h5_sho_targ_grp = make_group(h5_file, h5_sho_targ_grp)


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
            parms_dict = parms_dict = sidpy.hdf_utils.get_attributes(
                h5_main.parent.parent)

            print(
                f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters")

    @property
    def spectroscopic_values(self):
        """Spectroscopic values"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Spectroscopic_Values"][:]

    @property
    def be_repeats(self):
        """Number of BE repeats"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000'].attrs["BE_repeats"]

    @property
    def num_bins(self):
        """Number of frequency bins in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_bins"]

    @property
    def dc_voltage(self):
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["/Raw_Data-SHO_Fit_000/Spectroscopic_Values"][0, 1::2]

    @property
    def num_pix(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_pix"]
        
    @property
    def num_cycles(self):
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["VS_number_of_cycles"]

    @property
    def num_pix_1d(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return int(np.sqrt(self.num_pix))

    @property
    def voltage_steps(self):
        """Number of DC voltage steps"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_udvs_steps"]

    @property
    def sampling_rate(self):
        """Sampling rate in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["IO_rate_[Hz]"]

    @property
    def be_bandwidth(self):
        """BE bandwidth in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_band_width_[Hz]"]

    @property
    def be_center_frequency(self):
        """BE center frequency in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_center_frequency_[Hz]"]

    @property
    def frequency_bin(self):
        """Frequency bin vector in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Bin_Frequencies"][:]

    @property
    def be_waveform(self):
        """BE excitation waveform"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Excitation_Waveform"][:]

    @property
    def hysteresis_waveform(self, loop_number=2):
        with h5py.File(self.file, "r+") as h5_f:
            return (
                self.spectroscopic_values[1, ::len(self.frequency_bin)][int(self.voltage_steps/loop_number):] *
                self.spectroscopic_values[2, ::len(
                    self.frequency_bin)][int(self.voltage_steps/loop_number):]
            )

    @property
    def resampled_freq(self):
        return resample(self.frequency_bin, self.resampled_bins)

    # raw_be_data as complex
    @property
    def original_data(self):
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Raw_Data"][:]

    def raw_data(self, pixel=None, voltage_step=None, noise = None):
        """Raw data"""
        if pixel is not None and voltage_step is not None:
            with h5py.File(self.file, "r+") as h5_f:
                return self.raw_data_reshaped[[pixel], :, :][:, [voltage_step], :]
        else:
            with h5py.File(self.file, "r+") as h5_f:
                return self.raw_data_reshaped[:]

    def set_raw_data(self):
        with h5py.File(self.file, "r+") as h5_f:
            self.raw_data_reshaped = self.original_data.reshape(self.num_pix, self.voltage_steps, self.num_bins)
            # self.data_writer("Measurement_000/Channel_000", "Raw_Data_Reshaped",
            #                  self.original_data.reshape(self.num_pix, self.voltage_steps, self.num_bins))

    def SHO_Scaler(self,
                   save_loc='SHO_LSQF_scaled',
                   dataset="Raw_Data-SHO_Fit_000"):

        self.SHO_scaler = StandardScaler()
        data = self.SHO_LSQF(dataset).reshape(-1, 4)

        self.SHO_scaler.fit(data)

        # sets the phase not to scale
        self.SHO_scaler.mean_[3] = 0
        self.SHO_scaler.var_[3] = 1
        self.SHO_scaler.scale_[3] = 1


    def SHO_LSQF(self, data = "Raw_Data-SHO_Fit_000", pixel=None, voltage_step=None):
        with h5py.File(self.file, "r+") as h5_f:
            # dataset_ = h5_f['/Raw_Data-SHO_Fit_000/SHO_LSQF']
            dataset_ = self.SHO_LSQF_data[data]

            if pixel is not None and voltage_step is not None:
                return dataset_[[pixel], :, :][:, [voltage_step], :]
            elif pixel is not None:
                return dataset_[[pixel], :, :]
            else:
                return dataset_[:]

    @staticmethod
    def is_complex(data):
        data = data[0]

        if type(data) == torch.Tensor:
            complex_ = data.is_complex()

        if type(data) == np.ndarray:
            complex_ = np.iscomplex(data)
            complex_ = complex_.any()

        return complex_

    @staticmethod
    def to_magnitude(data):
        data = BE_Dataset.to_complex(data)
        return [np.abs(data), np.angle(data)]

    @staticmethod
    def to_real_imag(data):
        data = BE_Dataset.to_complex(data)
        return [np.real(data), np.imag(data)]

    @staticmethod
    def to_complex(data, axis = None):
        if type(data) == list:
            data = np.array(data)

        if BE_Dataset.is_complex(data):
            return data

        if type(data) == list:
            data = np.array(data)
        
        if axis is not None:
            pass
        else:
            axis = data.ndim - 1
    
        # elif data.ndim == 1:
        #     return data[0] + 1j * data[1]
        # elif data.ndim == 2:
        #     return data[:, 0] + 1j * data[:, 1]
        # elif data.ndim == 3:
        #     return data[:, :, 0] + 1j * data[:, :, 1]
        return np.take(data, 0, axis = axis) + 1j * np.take(data, 1, axis = axis)

    def set_SHO_LSQF(self, 
                     scaler="Raw_Data-SHO_Fit_000", 
                     save_loc='SHO_LSQF'):
        """Utility function to convert the SHO fit results to an array

        Args:
            SHO_LSQF (h5 Dataset): Location of the fit results in an h5 file

        Returns:
            np.array: SHO fit results
        """
        
        # data groups in file
        SHO_fits = find_groups_with_string(self.file, 'Raw_Data-SHO_Fit_000')
        
        # initializes the dictionary
        self.SHO_LSQF_data = {}
                
        with h5py.File(self.file, "r+") as h5_f:
            
            # loops around the found SHO_fits
            for SHO_fit in SHO_fits:
                
                # extract the name of the fit
                name = SHO_fit.split('/')[1]
            
                # create a list for parameters
                SHO_LSQF_list = []
                for sublist in np.array(
                    h5_f[f'{SHO_fit}/Fit']
                ):
                    for item in sublist:
                        for i in item:
                            SHO_LSQF_list.append(i)

                data_ = np.array(SHO_LSQF_list).reshape(
                    -1, 5)

                # # writes all but the r2
                # self.data_writer(
                #     basepath, save_loc, data_.reshape(
                #         self.num_pix, self.voltage_steps, 5)[:, :, :-1])
                
                self.SHO_LSQF_data[name] = data_.reshape(
                                self.num_pix, self.voltage_steps, 5)[:, :, :-1]
                
                # computes the scaler of the data
                if name == scaler:
                    self.SHO_Scaler(dataset = name)

    @staticmethod
    def shift_phase(phase, shift_=None):

        if shift_ is None:
            return phase
        else:
            shift = shift_

        if shift > 0:
            phase_ = phase
            phase_ += np.pi
            phase_[phase_ <= shift] += 2 *\
                np.pi  # shift phase values greater than pi
            return phase_ - shift - np.pi
        else:
            phase_ = phase
            phase_ -= np.pi
            phase_[phase_ >= shift] -= 2 *\
                np.pi  # shift phase values greater than pi
            return phase_ - shift + np.pi
        
        # removed copy TODO delete

    def raw_data_resampled(self, pixel=None, voltage_step=None, noise = None):
        """Resampled real part of the complex data resampled"""
        if pixel is not None and voltage_step is not None:
            return self.resampled_data["raw_data_resampled"][[pixel], :, :][:, [voltage_step], :]
            # with h5py.File(self.file, "r+") as h5_f:
            #     return h5_f[
            #         "Measurement_000/Channel_000/raw_data_resampled"][[pixel], :, :][:, [voltage_step], :]
        else:
            with h5py.File(self.file, "r+") as h5_f:
                return self.resampled_data["raw_data_resampled"][:]
            
                #h5_f[
                #    "Measurement_000/Channel_000/raw_data_resampled"][:]

    def measurement_state_voltage(self, voltage_step):
        """determines the pixel value based on the measurement state

        Args:
            voltage_step (int): voltage_step position

        Returns:
            voltage_step (int): pixel value in the correct state
        """
        if voltage_step is not None:

            # changes the pixel to collect the value for the on or off state
            if self.measurement_state == 'on':
                voltage_step = np.arange(0, self.voltage_steps)[
                    1::2][voltage_step]
            elif self.measurement_state == 'off':
                voltage_step = np.arange(0, self.voltage_steps)[
                    ::2][voltage_step]

        return voltage_step

    def SHO_fit_results(self, 
                        pixel=None, 
                        voltage_step=None, dataset = 'Raw_Data-SHO_Fit_000'):
        """Fit results"""
        with h5py.File(self.file, "r+") as h5_f:

            voltage_step = self.measurement_state_voltage(voltage_step)

            data = eval(f"self.SHO_{self.fitter}_data[dataset](pixel, voltage_step)")

            data_shape = data.shape

            data = data.reshape(-1, 4)

            if eval(f"self.{self.fitter}_phase_shift") is not None:
                data[:, 3] = eval(
                    f"self.shift_phase(data[:, 3], self.{self.fitter}_phase_shift)")

            data = data.reshape(data_shape)

            # does not sample if just a pixel is returned
            if pixel is None or voltage_step is None:
                data = self.get_voltage_state(data)

            if self.scaled:
                data = self.SHO_scaler.transform(
                    data.reshape(-1, 4)).reshape(data_shape)
            
            # reshapes the data to be (index, SHO_params)    
            if self.output_shape == "index":
                return data.reshape(-1, 4)

            return data
        
    def get_voltage_state(self, data):
        """function to get the voltage state of the data

        Args:
            data (any): BE data

        Returns:
            any: data with just the selected voltage state
        """                
        # only does this if getting the full dataset, will reduce to off and on state
        if self.measurement_state == 'all':
            data = data
        elif self.measurement_state == 'on':
            data = data[:, 1::2, :]
        elif self.measurement_state == 'off':
            data = data[:, ::2, :]
            
        return data
    
    def get_cycle(self, data, axis = 0,  **kwargs):
        data = np.array_split(data, self.num_cycles, axis = axis, **kwargs)
        data = data[self.cycle -1]
        return data 
            
    def get_measurement_cycle(self, data, cycle = None, axis = 1):
        if cycle is not None: 
            self.cycle = cycle
        data = self.get_voltage_state(data)
        return self.get_cycle(data, axis = axis)
    
    def get_raw_data_from_LSQF_SHO(self, model, index = None):
        # holds the raw state
        current_state = self.get_state
        
        self.scaled = False
        
        params_shifted = self.SHO_fit_results()
            
        exec(f"self.{model['fitter']}_phase_shift =0")
        
        params = self.SHO_fit_results()
        
        self.scaled = True
        
        pred_data = self.raw_spectra(
                fit_results=params)
        
        # output (channels, samples, voltage steps)
        pred_data = np.array([pred_data[0], pred_data[1]])
        
        # output (samples, channels, voltage steps)
        pred_data = np.swapaxes(pred_data, 0, 1)
        
        # output (samples, voltage steps, channels)
        pred_data = np.swapaxes(pred_data, 1, 2)
        
        if index is not None:
            pred_data = pred_data[[index]]
            params = params_shifted[[index]]
        
        self.set_attributes(**current_state)
        
        return pred_data, params
    
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def raw_spectra(self, 
                    pixel=None, 
                    voltage_step=None, 
                    fit_results=None, 
                    type_="numpy", 
                    frequency=False,
                    noise = None):
        """Raw spectra"""
        
        
        with h5py.File(self.file, "r+") as h5_f:
            
            # sets the shaper_ equal to true to correct the shape
            shaper_=True

            # gets the voltage steps to consider given the voltage state
            voltage_step = self.measurement_state_voltage(voltage_step)

            # if to get the resampled data
            if self.resampled:
                
                # get the number of bins to resample
                bins = self.resample_bins
                
                # gets the frequency values based on the resampled bins
                frequency_bins = self.get_freq_values(bins)
            else:
                
                # gets the unresampled bins
                bins = self.num_bins
                
                # gets the raw frequency bins
                frequency_bins = self.get_freq_values(bins)

            # if a fit_result is not provided get the raw data
            if fit_results is None:
                
                # gets the resampled data if resampled is set to true
                if self.resampled:
                    
                    # gets the data
                    data = self.raw_data_resampled(
                        pixel=pixel, voltage_step=voltage_step, noise = noise)

                else:
                    
                    # if not resampled gets the raw data
                    data = self.raw_data(
                        pixel=pixel, voltage_step=voltage_step, noise = noise)

            else:
                
                # if a fit result is provided gets the shape of the parameters
                params_shape = fit_results.shape

                # reshapes the parameters for fitting functions
                params = torch.tensor(fit_results.reshape(-1, 4))

                # gets the data from the fitting function
                data = eval(
                    f"self.SHO_fit_func_{self.fitter}(params, frequency_bins)")
                
                # checks if the full dataset was used and thus the data can be reshaped
                if bins*self.num_pix*self.voltage_steps*2 == len(data.flatten()):
                    pass
                else: 
                    shaper_ = False

                if shaper_:
                    data = self.shaper(data, pixel, voltage_step)

            if shaper_:
                # does not sample if just a pixel is returned
                if pixel is None or voltage_step is None:
                    data = self.get_voltage_state(data)

            if self.raw_format == 'complex':
                # computes the scaler on the raw data
                if self.scaled:
                    data = self.raw_data_scaler.transform(
                        data.reshape(-1, bins))
                
                if shaper_:
                    data = self.shaper(data, pixel, voltage_step)

                data = [np.real(data), np.imag(data)]
                
            elif self.raw_format == "magnitude spectrum":

                if shaper_:
                    data = self.shaper(data, pixel, voltage_step)

                data = [np.abs(data), np.angle(data)]

            # if a tensor converts to a numpy array
            try:
                data[0] = data[0].numpy()
                data[1] = data[1].numpy()
            except:
                pass

            if frequency:
                return data, frequency_bins
            else:
                return data

    def get_freq_values(self, data):

        try:
            data = data.flatten()
        except:
            pass
        
        
        if np.isscalar(data) or len(data) == 1:
            length = data
        else: 
            length = len(data)
        
        if length == self.resampled_bins:
            x = resample(self.frequency_bin,
                         self.resampled_bins)
        elif length == self.num_bins:
            x = self.frequency_bin
        else:
            raise ValueError(
                "original data must be the same length as the frequency bins or the resampled frequency bins")
        return x

    def shaper(self, data, pixel = None, voltage_steps = None, length = None ):
        
        # reshapes if you just grab a pixel.
        if pixel is not None:
            try:
                num_pix = len(pixel)
            except: 
                num_pix = 1
        else:
            num_pix = self.num_pix
            
        if voltage_steps is not None:
            try:                 
                voltage_steps = len(voltage_steps)
            except:
                voltage_steps = 1 
        else:
            voltage_steps = self.voltage_steps
        
        """Reshapes the data to the correct output shape"""
        if self.output_shape == "pixels":
            data = data.reshape(num_pix, voltage_steps, -1)
        elif self.output_shape == "index":
            data = data.reshape(num_pix * voltage_steps, -1)
        else:
            raise ValueError("output_shape must be either 'pixel' or 'index'")
        return data

    def set_raw_data_resampler(self,
                               save_loc='raw_data_resampled',
                               **kwargs):
        with h5py.File(self.file, "r+") as h5_f:
            if self.resampled_bins != self.num_bins:
                resampled_ = self.resampler(
                    self.raw_data().reshape(-1, self.num_bins), axis=2)
                self.resampled_data[save_loc] = resampled_
            else: 
                self.resampled_data[save_loc] = self.raw_data().reshape(-1, self.num_bins)
            
            if kwargs.get("basepath"):
                self.data_writer(kwargs.get("basepath"), save_loc, resampled_)

    def resampler(self, data, axis=2):
        """Resample the data to a given number of bins"""
        with h5py.File(self.file, "r+") as h5_f:
            try:
                return resample(data.reshape(self.num_pix, -1, self.num_bins),
                                self.resample_bins, axis=axis)
            except ValueError:
                print("Resampling failed, check that the number of bins is defined")

    @property
    def extraction_state(self):
        print(f'''
                  Resample = {self.resampled}
                  Raw Format = {self.raw_format}
                  fitter = {self.fitter}
                  scaled = {self.scaled}
                  Output Shape = {self.output_shape}
                  Measurement State = {self.measurement_state}
                  Resample Resampled = {self.resampled}
                  Resample Bins = {self.resample_bins}
                  LSQF Phase Shift = {self.LSQF_phase_shift}
                  NN Phase Shift = {self.NN_phase_shift}
                  Noise Level = {self.noise}
                  ''')

    @property
    def get_state(self):
        return {'resampled': self.resampled,
                'raw_format': self.raw_format,
                'fitter': self.fitter,
                'scaled': self.scaled,
                'output_shape': self.output_shape,
                'measurement_state': self.measurement_state,
                'resampled': self.resampled,
                'resample_bins': self.resample_bins,
                'LSQF_phase_shift': self.LSQF_phase_shift,
                'NN_phase_shift': self.NN_phase_shift}

    def NN_data(self, resampled=True, scaled=True):

        # makes sure you are using the resampled data
        self.resampled = resampled

        # makes sure you are using the scaled data
        self.scaled = scaled

        # gets the raw spectra
        data = self.raw_spectra()

        x_data = self.to_nn(data)

        # gets the SHO fit results these values are scaled
        y_data = self.SHO_fit_results().reshape(-1, 4)

        y_data = torch.tensor(y_data, dtype=torch.float32)

        return x_data, y_data

    def to_nn(self, data):
        
        if type(data) == torch.Tensor:
            return data
        
        if self.resampled:
            bins = self.resample_bins
        else: 
            bins = self.num_bins    

        real, imag = data

        # reshapes the data to be samples x timesteps
        real = real.reshape(-1, bins)
        imag = imag.reshape(-1, bins)

        # stacks the real and imaginary components
        x_data = np.stack((real, imag), axis=2)

        x_data = torch.tensor(x_data, dtype=torch.float32)

        return x_data

    def test_train_split_(self, test_size=0.2, random_state=42, resampled=True, scaled=True, shuffle=False):

        x_data, y_data = self.NN_data(resampled, scaled)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=shuffle)

        if self.verbose:
            self.extraction_state

        return self.X_train, self.X_test, self.y_train, self.y_test

    class Raw_Data_Scaler():

        def __init__(self, raw_data):
            self.raw_data = raw_data
            self.fit()

        @staticmethod
        def data_type_converter(data):

            if BE_Dataset.is_complex(data):
                return data
            else:
                return BE_Dataset.to_complex(data)

        def fit(self):
            data = self.raw_data

            data = self.data_type_converter(data)

            real = np.real(data)
            imag = np.imag(data)
            self.real_scaler = global_scaler()
            self.imag_scaler = global_scaler()

            self.real_scaler.fit(real)
            self.imag_scaler.fit(imag)

        def transform(self, data):

            data = self.data_type_converter(data)

            real = np.real(data)
            imag = np.imag(data)

            real = self.real_scaler.transform(real)
            imag = self.imag_scaler.transform(imag)

            return real + 1j*imag

        def inverse_transform(self, data):
            data = self.data_type_converter(data)

            real = np.real(data)
            imag = np.imag(data)

            real = self.real_scaler.inverse_transform(real)
            imag = self.imag_scaler.inverse_transform(imag)

            return real + 1j*imag
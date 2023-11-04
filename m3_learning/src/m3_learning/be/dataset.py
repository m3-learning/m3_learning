from m3_learning.util.h5_util import print_tree, get_tree
from m3_learning.util.rand_util import in_list
from BGlib import be as belib
import pyUSID as usid
import os
import sidpy
import numpy as np
import h5py
import time
from m3_learning.util.rand_util import extract_number
from m3_learning.util.h5_util import make_dataset, make_group, find_groups_with_string, find_measurement
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
from m3_learning.be.USID_data import USIDataset
from sidpy.hdf.hdf_utils import get_attr
from pyUSID.io.hdf_utils import check_if_main, create_results_group, write_reduced_anc_dsets, link_as_main, \
    get_dimensionality, get_sort_order, get_unit_values, reshape_to_n_dims, write_main_dataset, reshape_from_n_dims
from pyUSID.io.hdf_utils import reshape_to_n_dims, get_auxiliary_datasets
from m3_learning.be.filters import clean_interpolate
from dataclasses import dataclass, field, InitVar
from typing import Any, Callable, Dict, Optional, Union


def static_state_decorator(func):
    """Decorator that stops the function from changing the state

    Args:
        func (method): any method
    """
    def wrapper(*args, **kwargs):

        # saves the current state
        current_state = args[0].get_state

        # runs the function
        out = func(*args, **kwargs)

        # resets the state
        args[0].set_attributes(**current_state)

        # returns the output
        return out

    # returns the wrapper
    return wrapper


def resample(y, num_points, axis=0):
    """
    resample function to resample the data

    Args:
        y (np.array): data to resample
        num_points (int): number of points to resample
        axis (int, optional): axis to apply resampling. Defaults to 0.
    """

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


@dataclass
class BE_Dataset:
    file: str
    scaled: bool = False
    raw_format: str = "complex"
    fitter: str = 'LSQF'
    output_shape: str = 'pixels'
    measurement_state: str = 'all'
    resampled: bool = False
    resampled_bins: Optional[int] = field(default=None, init=False)
    LSQF_phase_shift: Optional[float] = None
    NN_phase_shift: Optional[float] = None
    verbose: bool = False
    noise_state: int = 0
    cleaned: bool = False
    basegroup: str = '/Measurement_000/Channel_000'
    SHO_fit_func_LSQF: Callable = field(default=SHO_fit_func_nn)
    hysteresis_function: Optional[Callable] = None
    loop_interpolated: bool = False
    tree: Any = field(init=False)
    resampled_data: Dict[str, Any] = field(default_factory=dict, init=False)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    """A class to represent a Band Excitation (BE) dataset.

    Attributes:
        file (str): The file path of the dataset.
        scaled (bool, optional): Whether the data is scaled. Defaults to False.
        raw_format (str, optional): The raw data format. Defaults to "complex".
        fitter (str, optional): The fitter to be used. Defaults to 'LSQF'.
        output_shape (str, optional): The output shape of the dataset. pixels is 2d, index is 1d Defaults to 'pixels'.
        measurement_state (str, optional): The state of the measurement. Defaults to 'all'.
        resampled (bool, optional): Whether the data is resampled. Defaults to False.
        resampled_bins (int, optional): The number of bins for resampling. Automatically set if None.
        LSQF_phase_shift (float, optional): The phase shift for LSQF.
        NN_phase_shift (float, optional): The phase shift for Neural Network.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
        noise_state (int, optional): Noise level. Defaults to 0.
        cleaned (bool, optional): Whether the data is cleaned. Defaults to False.
        basegroup (str, optional): The base group in the HDF5 file. Defaults to '/Measurement_000/Channel_000'.
        SHO_fit_func_LSQF (Callable, optional): The fitting function for SHO in NN.
        hysteresis_function (Callable, optional): The hysteresis function for processing. 
        loop_interpolated (bool, optional): Whether the loop data is interpolated. Defaults to False.
        resampled_data (Dict[str, Any]): Holds resampled data. Initialized post object creation.
        kwargs (Dict[str, Any], optional): Additional keyword arguments.
    """

    def __post_init__(self):
        self.noise = self.noise_state
        self.tree = self.get_tree()

        # Initialize resampled_bins if it's None
        if self.resampled_bins is None:
            self.resampled_bins = self.num_bins

        # Set additional attributes from kwargs
        for key, value in self.kwargs.items():
            setattr(self, key, value)

        # Preprocessing and raw data
        self.set_preprocessing()
        self.set_raw_data()
        self.SHO_preprocessing()

    def generate_noisy_data_records(self,
                                    noise_levels,
                                    basegroup='/Measurement_000/Channel_000',
                                    verbose=False,
                                    noise_STD=None):
        """
        generate_noisy_data_records Function that generates noisy data records and saves them to the H5 file

        Args:
            noise_levels (list): list of noise levels to be applied to the dataset
            basegroup (str, optional): basegroup where the data will be saved. Defaults to '/Measurement_000/Channel_000'.
            verbose (bool, optional): sets the verbosity of the function. Defaults to False.
            noise_STD (float, optional): manually provides a standard deviation value for the noise. Defaults to None.
        """

        # computes the noise state if a value is not provided
        if noise_STD is None:
            noise_STD = np.std(self.get_original_data)

        if verbose:
            print(f"The STD of the data is: {noise_STD}")

        with h5py.File(self.file, "r+") as h5_f:

            # iterates through the noise levels provided
            for noise_level in noise_levels:

                if verbose:
                    print(f"Adding noise level {noise_level}")

                # computes the noise level
                noise_level_ = noise_STD * noise_level

                # computes the real and imaginary components of the noise
                noise_real = np.random.uniform(-1*noise_level_,
                                               noise_level_, (self.num_pix, self.spectroscopic_length))
                noise_imag = np.random.uniform(-1*noise_level_,
                                               noise_level_, (self.num_pix, self.spectroscopic_length))

                # adds the noise to the original data
                noise = noise_real+noise_imag*1.0j
                data = self.get_original_data + noise

                h5_main = usid.hdf_utils.find_dataset(h5_f, "Raw_Data")[0]

                # writes the noise record to the pyUSID file
                usid.hdf_utils.write_main_dataset(h5_f[basegroup],  # parent group
                                                  data,  # data to be written
                                                  # Name of the main dataset
                                                  f'Noisy_Data_{noise_level}',
                                                  'Piezoresponse',  # quantity
                                                  'V',  # units
                                                  None,  # position dimensions
                                                  None,  # spectroscopic dimensions
                                                  pos_ind=h5_main.pos_ind,
                                                  h5_pos_vals=h5_main.h5_pos_vals,
                                                  h5_spec_inds=h5_main.h5_spec_inds,
                                                  h5_spec_vals=h5_main.h5_spec_vals,
                                                  compression='gzip')

    def set_noise_state(self, noise):
        """function that uses the noise state to set the current dataset

        Args:
            noise (int): noise value in multiples of the standard deviation

        Raises:
            ValueError: error if the noise value does not exist in the dataset
        """

        if noise == 0:
            self.dataset = "Raw_Data"
        else:
            self.dataset = f"Noisy_Data_{noise}"

    def set_preprocessing(self):
        """
        set_preprocessing searches the dataset to see what preprocessing is required.
        """

        # does preprocessing for the SHO_fit results
        if in_list(self.tree, "*SHO_Fit*"):
            self.SHO_preprocessing()
        else:
            Warning("No SHO fit found")

        # does preprocessing for the loop fit results
        if in_list(self.tree, "*Fit-Loop_Fit*"):
            self.loop_fit_preprocessing()

    def loop_fit_preprocessing(self):
        """
        loop_fit_preprocessing preprocessing for the loop fit results
        """

        # gets the hysteresis loops
        hysteresis, bias = self.get_hysteresis(
            plotting_values=True, output_shape="index")

        # interpolates any missing points in the data
        cleaned_hysteresis = clean_interpolate(hysteresis)

        # instantiates and computes the global scaler
        self.hystersis_scaler = global_scaler()
        self.hystersis_scaler.fit_transform(cleaned_hysteresis)
        
        try:
            self.LoopParmScaler()
        except:
            pass

    @property
    def hysteresis_scaler(self):
        """
        get_hysteresis_scaler gets the hysteresis scaler

        Returns:
            scaler: scaler for the hysteresis loops
        """

        return self.hystersis_scaler
    
    @property
    def get_voltage(self):
        """
        get_voltage gets the voltage vector

        Returns:
            np.array: voltage vector
        """
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000']['Channel_000']['UDVS'][::2][:, 1][24:120] * -1

        

    def SHO_preprocessing(self):
        """
        SHO_preprocessing conducts the preprocessing on the SHO fit results
        """

        # extract the raw data and reshapes is
        self.set_raw_data()

        # resamples the data if necessary
        self.set_raw_data_resampler()

        # computes the scalar on the raw data
        self.raw_data_scaler = self.Raw_Data_Scaler(self.raw_data())

        try:
            # gets the LSQF results
            self.set_SHO_LSQF()

            # computes the SHO scaler
            self.SHO_Scaler()
        except:
            pass

    def default_state(self):
        """
        default_state Function that returns the dataset to the default state
        """

        # dictionary of the default state
        default_state_ = {'raw_format': "complex",
                          "fitter": 'LSQF',
                          "output_shape": "pixels",
                          "scaled": False,
                          "measurement_state": "all",
                          "resampled": False,
                          "resampled_bins": 80,
                          "LSQF_phase_shift": None,
                          "NN_phase_shift": None, }

        # sets the atributes to the default state
        self.set_attributes(**default_state_)

    def get_tree(self):
        """
        get_tree reads the tree from the H5 file

        Returns:
            list: list of the tree from the H5 file
        """

        with h5py.File(self.file, "r+") as h5_f:
            return get_tree(h5_f)

    def print_be_tree(self):
        """Utility file to print the Tree of a BE Dataset

        Code adapted from pyUSID

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
        """
        data_writer function to write data to an USID dataset

        Args:
            base (str): basepath where to save the data
            name (str): name of the dataset to save
            data (np.array): data to save
        """

        with h5py.File(self.file, "r+") as h5_f:

            try:
                # if the dataset does not exist can write
                make_dataset(h5_f[base],
                             name,
                             data)

            except:
                # if the dataset exists deletes the dataset and then writes
                self.delete(f"{base}/{name}")
                make_dataset(h5_f[base],
                             name,
                             data)

    # delete a dataset
    def delete(self, name):
        """
        delete function to delete a dataset within a pyUSID file

        Args:
            name (str): path of dataset to delete
        """

        with h5py.File(self.file, "r+") as h5_f:
            try:
                del h5_f[name]
            except KeyError:
                print("Dataset not found, could not be deleted")

    def SHO_Fitter(self, force=False, max_cores=-1, max_mem=1024*8,
                   dataset="Raw_Data",
                   h5_sho_targ_grp=None):
        """Function that computes the SHO fit results

        This function is adapted from BGlib

        Args:
            force (bool, optional): forces the SHO results to be computed from scratch. Defaults to False.
            max_cores (int, optional): number of processor cores to use. Defaults to -1.
            max_mem (_type_, optional): maximum ram to use. Defaults to 1024*8.
        """

        with h5py.File(self.file, "r+") as h5_file:

            # the start time of the fit
            start_time_lsqf = time.time()

            # splits the directory path and the file name
            (data_dir, filename) = os.path.split(self.file)

            if self.file.endswith(".h5"):
                # No translation here
                h5_path = self.file

            else:
                pass

            # splits the path and the folder name
            folder_path, h5_raw_file_name = os.path.split(h5_path)

            # h5_file = h5py.File(h5_path, "r+")
            print("Working on:\n" + h5_path)

            # get the main dataset
            h5_main = usid.hdf_utils.find_dataset(h5_file, dataset)[0]

            # grabs some useful parameters from the dataset
            pos_ind = h5_main.h5_pos_inds
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
                    print(
                        "VS cycle fraction could not be found. Setting to default value")
                    vs_cycle_frac = "full"

            sho_fit_points = 5  # The number of data points at each step to use when fitting
            sho_override = force  # Force recompute if True

            # h5_sho_targ_grp = None
            h5_sho_file_path = os.path.join(
                folder_path, h5_raw_file_name)

            print("\n\nSHO Fits will be written to:\n" +
                  h5_sho_file_path + "\n\n")
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
            parms_dict = sidpy.hdf_utils.get_attributes(
                h5_main.parent.parent)

            print(
                f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters")

            return sho_fitter

    def measure_group(self):
        """
        measure_group gets the measurement group based on a noise leve

        Returns:
            str: string for the measurment group for the data
        """

        if self.noise == 0:
            return "Raw_Data_SHO_Fit"
        else:
            return f"Noisy_Data_{self.noise}"

    def LSQF_Loop_Fit(self,
                      main_dataset='Raw_Data-SHO_Fit_000/Fit',
                      h5_target_group=None,
                      max_cores=None):
        """
        LSQF_Loop_Fit Function that conducts the hysteresis loop fits based on the LSQF results. 

        This is adapted from BGlib

        Args:
            main_dataset (str, optional): main dataset where loop fits are conducted from. Defaults to 'Raw_Data-SHO_Fit_000/Fit'.
            h5_target_group (str, optional): path where the data will be saved to. Defaults to None.
            max_cores (int, optional): number of cores the fitter will use, -1 will use all cores. Defaults to None.

        Raises:
            TypeError: _description_

        Returns:
            tuple: results from the loop fit, group where the loop fit is
        """

        with h5py.File(self.file, "r+") as h5_file:

            # gets the experiment type from the file
            expt_type = sidpy.hdf.hdf_utils.get_attr(h5_file, 'data_type')

            # finds the dataset from the file
            h5_meas_grp = usid.hdf_utils.find_dataset(
                h5_file, self.measure_group())

            # extract the voltage mode
            vs_mode = sidpy.hdf.hdf_utils.get_attr(
                h5_file["/Measurement_000"], 'VS_mode')

            try:
                vs_cycle_frac = sidpy.hdf.hdf_utils.get_attr(
                    h5_file["/Measurement_000"], 'VS_cycle_fraction')

            except KeyError:
                print('VS cycle fraction could not be found. Setting to default value')
                vs_cycle_frac = 'full'

            if isinstance(main_dataset, str):
                main_dataset = USIDataset(h5_file[main_dataset])
            elif isinstance(main_dataset, USIDataset):
                pass
            else:
                raise TypeError(
                    'main_dataset should be a string or USIDataset object')

            # instantiates the loopfitter using belib
            loop_fitter = belib.analysis.BELoopFitter(main_dataset,
                                                      expt_type, vs_mode, vs_cycle_frac,
                                                      h5_target_group=h5_target_group,
                                                      cores=max_cores,
                                                      verbose=False)

            # computes the guess for the loop fits
            loop_fitter.set_up_guess()
            h5_loop_guess = loop_fitter.do_guess(override=False)

            # Calling explicitly here since Fitter won't do it automatically
            h5_guess_loop_parms = loop_fitter.extract_loop_parameters(
                h5_loop_guess)
            loop_fitter.set_up_fit()
            h5_loop_fit = loop_fitter.do_fit(override=False)

            # save the path where the loop fit results are saved
            h5_loop_group = h5_loop_fit.parent

        return h5_loop_fit, h5_loop_group

    @property
    def num_cols(self):
        """Number of columns in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000'].attrs["grid_num_cols"]

    @property
    def num_rows(self):
        """Number of rows in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000'].attrs["grid_num_rows"]

    @property
    def noise(self):
        """Noise value"""
        return self._noise

    @noise.setter
    def noise(self, noise):
        """Sets the noise value"""
        self._noise = noise
        self.set_noise_state(noise)

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
        """Gets the DC voltage vector"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[f"Raw_Data-SHO_Fit_000/Spectroscopic_Values"][0, 1::2]

    @property
    def num_pix(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_pix"]

    @property
    def num_cycles(self):
        '''Gets the number of cycles in the data'''
        with h5py.File(self.file, "r+") as h5_f:
            cycles = h5_f["Measurement_000"].attrs["VS_number_of_cycles"]

            if h5_f["Measurement_000"].attrs["VS_measure_in_field_loops"] == 'in and out-of-field':
                cycles *= 2

            return cycles

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
        """Gets the hysteresis waveform"""
        with h5py.File(self.file, "r+") as h5_f:
            return (
                self.spectroscopic_values[1, ::len(self.frequency_bin)][int(self.voltage_steps/loop_number):] *
                self.spectroscopic_values[2, ::len(
                    self.frequency_bin)][int(self.voltage_steps/loop_number):]
            )

    @property
    def resampled_freq(self):
        """Gets the resampled frequency"""
        return resample(self.frequency_bin, self.resampled_bins)

    @property
    def spectroscopic_length(self):
        """Gets the length of the spectroscopic vector"""
        return self.num_bins*self.voltage_steps

    @property
    def get_original_data(self):
        """
        get_original_data gets the raw BE data as a complex value

        Returns:
            np.array: BE data as a complex number
        """
        with h5py.File(self.file, "r+") as h5_f:
            if self.dataset == 'Raw_Data':
                return h5_f["Measurement_000"]["Channel_000"]["Raw_Data"][:]
            else:
                name = find_measurement(self.file,
                                        f"original_data_{self.noise}STD",
                                        group=self.basegroup)
                return h5_f["Measurement_000"]["Channel_000"][name][:]

    def raw_data(self, pixel=None, voltage_step=None, noise=None):
        """
        raw_data function that extracts the raw data with consideration of the noise. Will return the resampled data

        Args:
            pixel (int, optional): pixel position to get data. Defaults to None.
            voltage_step (int, optional): voltage position to get data. Defaults to None.
            noise (int, optional): Noise value. Defaults to None.

        Returns:
            np.array: BE data as a complex number
        """
        if pixel is not None and voltage_step is not None:
            with h5py.File(self.file, "r+") as h5_f:
                return self.raw_data_reshaped[self.dataset][[pixel], :, :][:, [voltage_step], :]
        else:
            with h5py.File(self.file, "r+") as h5_f:
                return self.raw_data_reshaped[self.dataset][:]

    @static_state_decorator
    def set_raw_data(self):
        """
        set_raw_data Function that parses the datafile and extracts the raw data names
        """

        with h5py.File(self.file, "r+") as h5_f:
            # initializes the dictionary
            self.raw_data_reshaped = {}

            # list of datasets to be read
            datasets = []
            self.raw_datasets = []

            # Finds all the datasets
            datasets.extend(usid.hdf_utils.find_dataset(
                h5_f['Measurement_000/Channel_000'], 'Noisy'))
            datasets.extend(usid.hdf_utils.find_dataset(
                h5_f['Measurement_000/Channel_000'], 'Raw_Data'))

            # loops around all the datasets and stores them reshaped in a dictionary
            for dataset in datasets:
                self.raw_data_reshaped[dataset.name.split(
                    '/')[-1]] = dataset[:].reshape(self.num_pix, self.voltage_steps, self.num_bins)

                self.raw_datasets.extend([dataset.name.split('/')[-1]])

    @static_state_decorator
    def LSQF_hysteresis_params(self, output_shape=None, scaled=None, measurement_state=None):
        """
        LSQF_hysteresis_params Gets the LSQF hysteresis parameters

        Args:
            output_shape (str, optional): pixel or list. Defaults to None.
            scaled (bool, optional): selects if to scale the data. Defaults to None.
            measurement_state (any, optional): sets the measurement state. Defaults to None.

        Returns:
            np.array: hysteresis loop parameters from LSQF
        """

        if measurement_state is not None:
            self.measurement_state = measurement_state

        # sets output shape if provided
        if output_shape is not None:
            self.output_shape = output_shape

        # sets data to be scaled is provided
        if scaled is not None:
            self.scaled = scaled

        # extracts the hysteresis parameters from the H5 file
        with h5py.File(self.file, "r+") as h5_f:
            data = h5_f[f"/{self.dataset}-SHO_Fit_000/Fit-Loop_Fit_000/Fit"][:]
            data = data.reshape(self.num_rows, self.num_cols, self.num_cycles)
            data = np.array([data['a_0'], data['a_1'], data['a_2'], data['a_3'], data['a_4'],
                            data['b_0'], data['b_1'], data['b_2'], data['b_3']]).transpose((1, 2, 3, 0))

            if self.scaled:
                # TODO: add the scaling here
                data = self.loop_param_scaler.fit(data)
                
                # Warning("Scaling not implemented yet")
                # pass

            if self.output_shape == "index":
                data = data.reshape(
                    self.num_pix, self.num_cycles, data.shape[-1])

            data = self.hysteresis_measurement_state(data)

            return data

    @static_state_decorator
    def SHO_Scaler(self,
                   noise=0):
        """
        SHO_Scaler SHO scaler function

        Args:
            noise (int, optional): noise level to apply the scaler. Defaults to 0.
        """

        # set the noise and the dataset
        self.noise = noise

        self.SHO_scaler = StandardScaler()
        data = self.SHO_LSQF().reshape(-1, 4)

        self.SHO_scaler.fit(data)

        # sets the phase not to scale
        self.SHO_scaler.mean_[3] = 0
        self.SHO_scaler.var_[3] = 1
        self.SHO_scaler.scale_[3] = 1
        
    def LoopParmScaler(self):
        
        self.loop_param_scaler = StandardScaler()
        data = self.LSQF_hysteresis_params().reshape(-1, 9)
        
        self.loop_param_scaler.fit(data)

    def SHO_LSQF(self, pixel=None, voltage_step=None):
        """
        SHO_LSQF Gets the least squares SHO fit results

        Args:
            pixel (int, optional): selected pixel index to extract. Defaults to None.
            voltage_step (int, optional): selected voltage index to extract. Defaults to None.

        Returns:
            np.array: SHO LSQF results
        """

        with h5py.File(self.file, "r+") as h5_f:

            dataset_ = self.SHO_LSQF_data[f"{self.dataset}-SHO_Fit_000"].copy()

            if pixel is not None and voltage_step is not None:
                return self.get_data_w_voltage_state(dataset_[[pixel], :, :])[:, [voltage_step], :]
            elif pixel is not None:
                return self.get_data_w_voltage_state(dataset_[[pixel], :, :])
            else:
                return self.get_data_w_voltage_state(dataset_[:])

    @staticmethod
    def is_complex(data):
        """
        is_complex function to check if data is complex. If not complex makes it a complex number

        Args:
            data (any): input data

        Returns:
            any: array or tensor as a complex number
        """

        data = data[0]

        if type(data) == torch.Tensor:
            complex_ = data.is_complex()

        if type(data) == np.ndarray:
            complex_ = np.iscomplex(data)
            complex_ = complex_.any()

        return complex_

    @staticmethod
    def to_magnitude(data):
        """
        to_magnitude converts a complex number to an amplitude and phase

        Args:
            data (np.array): complex photodiode response of the cantilver

        Returns:
            list: list of np.array containing the magnitude and phase of the cantilever response
        """
        data = BE_Dataset.to_complex(data)
        return [np.abs(data), np.angle(data)]

    @staticmethod
    def to_real_imag(data):
        """
        to_real_imag function to extract the real and imaginary data components

        Args:
            data (np.array or torch.Tensor): BE data

        Returns:
            list: a list of np.arrays representing the real and imaginary components of the BE response.
        """
        data = BE_Dataset.to_complex(data)
        return [np.real(data), np.imag(data)]

    @staticmethod
    def to_complex(data, axis=None):
        """
        to_complex function that converts data to complex

        Args:
            data (any): data to convert
            axis (int, optional): axis which the data is structured. Defaults to None.

        Returns:
            np.array: complex array of the BE response
        """

        # converts to an array
        if type(data) == list:
            data = np.array(data)

        # if the data is already in complex form return
        if BE_Dataset.is_complex(data):
            return data

        # if axis is not provided take the last axis
        if axis is None:
            axis = data.ndim - 1

        return np.take(data, 0, axis=axis) + 1j * np.take(data, 1, axis=axis)

    def set_SHO_LSQF(self):
        """
        set_SHO_LSQF Sets the SHO Scaler data to make accessible
        """

        # initializes the dictionary
        self.SHO_LSQF_data = {}

        for dataset in self.raw_datasets:

            # data groups in file
            SHO_fits = find_groups_with_string(
                self.file, f'{dataset}-SHO_Fit_000')[0]

            with h5py.File(self.file, "r+") as h5_f:

                # extract the name of the fit
                name = SHO_fits.split('/')[-1]

                # create a list for parameters
                SHO_LSQF_list = []
                for sublist in np.array(
                    h5_f[f'{SHO_fits}/Fit']
                ):
                    for item in sublist:
                        for i in item:
                            SHO_LSQF_list.append(i)

                data_ = np.array(SHO_LSQF_list).reshape(
                    -1, 5)

                # saves the SHO LSQF data as an attribute of the dataset object
                self.SHO_LSQF_data[name] = data_.reshape(
                    self.num_pix, self.voltage_steps, 5)[:, :, :-1]

    @staticmethod
    def shift_phase(phase, shift_=None):
        """
        shift_phase function that shifts the phase of the dataset

        Args:
            phase (np.array): phase data
            shift_ (float, optional): phase to shift the data in radians. Defaults to None.

        Returns:
            np.array: phase shifted data
        """

        if shift_ is None or shift_ == 0:
            return phase
        else:
            shift = shift_

        if shift > 0:
            phase_ = phase
            phase_ += np.pi
            phase_[phase_ <= shift] += 2 *\
                np.pi  # shift phase values greater than pi
            phase__ = phase_ - shift - np.pi
        else:
            phase_ = phase
            phase_ -= np.pi
            phase_[phase_ >= shift] -= 2 *\
                np.pi  # shift phase values greater than pi
            phase__ = phase_ - shift + np.pi

        return phase__

    def raw_data_resampled(self, pixel=None, voltage_step=None):
        """
        raw_data_resampled Resampled real part of the complex data resampled

        Args:
            pixel (int, optional): selected pixel of data to resample. Defaults to None.
            voltage_step (int, optional): selected voltage step of data to resample. Defaults to None.

        Returns:
            np.array: resampled data
        """

        if pixel is not None and voltage_step is not None:
            return self.resampled_data[self.dataset][[pixel], :, :][:, [voltage_step], :]
        else:
            with h5py.File(self.file, "r+") as h5_f:
                return self.resampled_data[self.dataset][:]

    def measurement_state_voltage(self, voltage_step):
        """
        measurement_state_voltage determines the pixel value based on the measurement state

        Args:
            voltage_step (int): voltage step to select

        Returns:
            np.array: voltage vector
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

    def state_num_voltage_steps(self):
        """
        state_num_voltage_steps gets the number of voltage steps given the current measurement state

        Returns:
            int: number of voltage steps
        """

        if self.measurement_state == 'all':
            voltage_step = self.voltage_steps
        else:
            voltage_step = int(self.voltage_steps/2)

        return voltage_step

    @static_state_decorator
    def SHO_fit_results(self,
                        state=None,
                        model=None,
                        phase_shift=None,
                        X_data=None):
        """
        SHO_fit_results general function to get the SHO fit results from the dataset

        Args:
            state (dict, optional): a provided measurement state. Defaults to None.
            model (nn.module, optional): model which to get the data from. Defaults to None.
            phase_shift (float, optional): value to shift the phase. Defaults to None.
            X_data (np.array, optional): frequency bins. Defaults to None.

        Returns:
            np.array: SHO fit parameters
        """

        # Note removed pixel and voltage step indexing here

        # if a neural network model is not provided use the LSQF
        if model is None:

            # reads the H5 file
            with h5py.File(self.file, "r+") as h5_f:

                # sets a state if it is provided as a dictionary
                if state is not None:
                    self.set_attributes(**state)

                data = eval(f"self.SHO_{self.fitter}()")

                data_shape = data.shape

                data = data.reshape(-1, 4)

                if eval(f"self.{self.fitter}_phase_shift") is not None and phase_shift is None:
                    data[:, 3] = eval(
                        f"self.shift_phase(data[:, 3], self.{self.fitter}_phase_shift)")

                data = data.reshape(data_shape)

                if self.scaled:
                    data = self.SHO_scaler.transform(
                        data.reshape(-1, 4)).reshape(data_shape)

        elif model is not None:

            if X_data is None:
                X_data, Y_data = self.NN_data()

            # you can view the test and training dataset by replacing X_data with X_test or X_train
            pred_data, scaled_param, data = model.predict(X_data)

            if self.scaled:
                data = scaled_param

        if phase_shift is not None:
            data[:, 3] = self.shift_phase(data[:, 3], phase_shift)

        # reshapes the data to be (index, SHO_params)
        if self.output_shape == "index":
            return data.reshape(-1, 4)
        else:
            return data.reshape(self.num_pix, self.state_num_voltage_steps(), 4)

    def get_data_w_voltage_state(self, data):
        """
        get_data_w_voltage_state function to extract data given a voltage state

        Args:
            data (np.array): BE data

        Returns:
            np.array: BE data considering the voltage state
        """

        # only does this if getting the full dataset, will reduce to off and on state
        if self.measurement_state == 'all':
            data = data
        elif self.measurement_state == 'on':
            data = data[:, 1::2, :]
        elif self.measurement_state == 'off':
            data = data[:, ::2, :]

        return data

    def get_cycle(self, data, axis=0,  **kwargs):
        """
        get_cycle gets data for a specific cycle of the hysteresis loop

        Args:
            data (np.array): band excitation data to extract cycle from
            axis (int, optional): axis to cut the data cycles. Defaults to 0.

        Returns:
            np.array: data for a specific cycle
        """
        data = np.array_split(data, self.num_cycles, axis=axis, **kwargs)
        data = data[self.cycle - 1]
        return data

    def get_measurement_cycle(self, data, cycle=None, axis=1):
        """
        get_measurement_cycle function to get the cycle of a measurement

        Args:
            data (np.array): band excitation data to extract cycle from
            cycle (int, optional): cycle to extract. Defaults to None.
            axis (int, optional): axis where the cycle dimension is located. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if cycle is not None:
            self.cycle = cycle
        data = self.get_data_w_voltage_state(data)
        return self.get_cycle(data, axis=axis)

    @static_state_decorator
    def get_raw_data_from_LSQF_SHO(self, model, index=None):
        """
        get_raw_data_from_LSQF_SHO Extracts the raw data from LSQF SHO fits

        Args:
            model (dict): dictionary that defines the state to extract
            index (int, optional): index to extract. Defaults to None.

        Returns:
            tuple: output results from LSQF reconstruction, SHO parameters
        """

        # sets the attribute state based on the dictionary
        self.set_attributes(**model)

        # sets to get the unscaled parameters
        # this is required so the reconstructions are correct
        self.scaled = False

        # gets the SHO results
        params_shifted = self.SHO_fit_results()

        # sets the phase shift for the current fitter = 0
        # this is a requirement so the computed results are not phase shifted
        exec(f"self.{model['fitter']}_phase_shift=0")

        # gets the SHO fit parameters
        params = self.SHO_fit_results()

        # changes the state back to scaled
        self.scaled = True

        # gets the raw spectra computed based on the parameters
        # the output is the scaled values
        pred_data = self.raw_spectra(
            fit_results=params)

        # builds an array of the amplitude and phase
        pred_data = np.array([pred_data[0], pred_data[1]])

        # reshapes the data to be consistent with the rest of the package
        pred_data = np.swapaxes(pred_data, 0, 1)
        pred_data = np.swapaxes(pred_data, 1, 2)

        if index is not None:
            pred_data = pred_data[[index]]
            params = params_shifted[[index]]

        return pred_data, params

    def set_attributes(self, **kwargs):
        """
        set_attributes sets attributes of the object from a dictionary
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        # if noise is included this calls the setter function
        if kwargs.get("noise"):
            self.noise = kwargs.get("noise")

    @static_state_decorator
    def raw_spectra(self,
                    pixel=None,
                    voltage_step=None,
                    fit_results=None,
                    frequency=False,
                    noise=None,
                    state=None):
        """
        raw_spectra Function that simplifies getting the raw band excitation data

        Args:
            pixel (int, optional): pixel value to get. Defaults to None.
            voltage_step (int, optional): voltage step to get. Defaults to None.
            fit_results (np.array, optional): provided fit results to get raw spectra with. Defaults to None.
            frequency (bool, optional): option to return the frequency bins. Defaults to False.
            noise (int, optional): noise level to extract . Defaults to None.
            state (dict, optional): dictionary that defines what data is extracted. Defaults to None.

        Returns:
            np.array: band excitation data (if frequency == True will return the frequency bins)
        """

        # set the noise
        if noise is not None:
            self.noise = noise

        # sets the state if it is provided
        if state is not None:
            self.set_attributes(**state)

        with h5py.File(self.file, "r+") as h5_f:

            # sets the shaper_ equal to true to correct the shape
            shaper_ = True

            # gets the voltage steps to consider given the voltage state
            voltage_step = self.measurement_state_voltage(voltage_step)

            # if to get the resampled data
            if self.resampled:

                # get the number of bins to resample
                bins = self.resampled_bins

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
                        pixel=pixel, voltage_step=voltage_step)

                else:

                    # if not resampled gets the raw data
                    data = self.raw_data(
                        pixel=pixel, voltage_step=voltage_step)

            else:

                # if a fit result is provided gets the shape of the parameters
                params_shape = fit_results.shape

                if isinstance(fit_results, np.ndarray):
                    fit_results = torch.from_numpy(fit_results)

                # reshapes the parameters for fitting functions
                params = fit_results.reshape(-1, 4)

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
                    data = self.get_data_w_voltage_state(data)

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
        """
        get_freq_values Function that gets the frequency bins

        Args:
            data (np.array): BE data

        Raises:
            ValueError: original data and frequency bin mismatch

        Returns:
            np.array: frequency bins for the data
        """

        try:
            data = data.flatten()
        except:
            pass

        if np.isscalar(data) or len(data) == 1:
            length = data
        else:
            length = len(data)

        # checks if the length of the data is the raw length, or the resampled length
        if length == self.num_bins:
            x = self.frequency_bin
        elif length == self.resampled_bins:
            x = resample(self.frequency_bin,
                         self.resampled_bins)
        else:
            raise ValueError(
                "original data must be the same length as the frequency bins or the resampled frequency bins")
        return x

    def shaper(self,
               data,
               pixel=None,
               voltage_steps=None):
        """
        shaper Utility to help reshape band excitation data based on the current measurement state

        Args:
            data (np.array): band excitation data
            pixel (int, optional): _description_. Defaults to None.
            voltage_steps (int, optional): _description_. Defaults to None.

        Raises:
            ValueError: Invalid output shape is provided

        Returns:
            np.array: reshaped BE data
        """

        # reshapes if you just grab a pixel.
        if pixel is not None:
            try:
                num_pix = len(pixel)
            except:
                num_pix = 1
        else:
            num_pix = int(self.num_pix.copy())

        if voltage_steps is not None:
            try:
                voltage_steps = len(voltage_steps)
            except:
                voltage_steps = 1
        else:
            # computes the number of voltage steps in the data
            voltage_steps = int(self.voltage_steps.copy())

            if self.measurement_state in ["on", "off"]:
                voltage_steps /= 2
                voltage_steps = int(voltage_steps)

        # reshapes the data to be the correct output shape
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
        """
        set_raw_data_resampler function to compute the resampled raw data and save it to the USID file. 

        Args:
            save_loc (str, optional): filepath where the resampled data should be saved. Defaults to 'raw_data_resampled'.
        """
        with h5py.File(self.file, "r+") as h5_f:
            if self.resampled_bins != self.num_bins:
                for data in self.raw_datasets:
                    # resamples the data
                    resampled_ = self.resampler(
                        self.raw_data_reshaped[data].reshape(-1, self.num_bins), axis=2)
                    self.resampled_data[data] = resampled_.reshape(
                        self.num_pix, self.voltage_steps, self.resampled_bins)
            else:
                self.resampled_data = self.raw_data_reshaped

            # writes the data within the basepath
            if kwargs.get("basepath"):
                self.data_writer(kwargs.get("basepath"), save_loc, resampled_)

    def resampler(self, data, axis=2):
        """
        resampler Resamples the data to a given number of bins

        Args:
            data (np.array): BE dataset
            axis (int, optional): axis which to resample along. Defaults to 2.

        Returns:
            np.array: resampled band excitation data
        """
        with h5py.File(self.file, "r+") as h5_f:
            try:
                return resample(data.reshape(self.num_pix, -1, self.num_bins),
                                self.resampled_bins, axis=axis)
            except ValueError:
                print("Resampling failed, check that the number of bins is defined")

    @property
    def extraction_state(self):
        """
        extraction_state Function that prints the current extraction state
        """
        print(f'''
        Dataset = {self.dataset}
        Resample = {self.resampled}
        Raw Format = {self.raw_format}
        fitter = {self.fitter}
        scaled = {self.scaled}
        Output Shape = {self.output_shape}
        Measurement State = {self.measurement_state}
        Resample Resampled = {self.resampled}
        Resample Bins = {self.resampled_bins}
        LSQF Phase Shift = {self.LSQF_phase_shift}
        NN Phase Shift = {self.NN_phase_shift}
        Noise Level = {self.noise}
        loop interpolated = {self.loop_interpolated}
                    ''')

    @property
    def get_state(self):
        """
        get_state function that return the dictionary of the current state

        Returns:
            dict: dictionary of the current state
        """
        return {'resampled': self.resampled,
                'raw_format': self.raw_format,
                'fitter': self.fitter,
                'scaled': self.scaled,
                'output_shape': self.output_shape,
                'measurement_state': self.measurement_state,
                'resampled': self.resampled,
                'resampled_bins': self.resampled_bins,
                'LSQF_phase_shift': self.LSQF_phase_shift,
                'NN_phase_shift': self.NN_phase_shift,
                "noise": self.noise,
                "loop_interpolated": self.loop_interpolated}

    @static_state_decorator
    def NN_data(self, resampled=None, scaled=True):
        """
        NN_data utility function that gets the neural network data

        Args:
            resampled (bool, optional): sets if you should use the resampled data. Defaults to None.
            scaled (bool, optional): sets if you should use the scaled data. Defaults to True.

        Returns:
            torch.tensor: neural network input, SHO LSQF fit parameters (scaled)
        """

        print(self.extraction_state)

        if resample is not None:

            # makes sure you are using the resampled data
            self.resampled = resampled

        # makes sure you are using the scaled data
        # this is a requirement of training a neural network
        self.scaled = scaled

        # gets the raw spectra
        data = self.raw_spectra()

        # converts data to the form for a neural network
        x_data = self.to_nn(data)

        # gets the SHO fit results these values are scaled
        # this is from the LSQF used for evaluation
        y_data = self.SHO_fit_results().reshape(-1, 4)

        # converts the LSQF into a tensor for comparison
        y_data = torch.tensor(y_data, dtype=torch.float32)

        return x_data, y_data

    def to_nn(self, data):
        """
        to_nn utility function that converts band excitation data into a form suitable for training a neural network

        Args:
            data (any): band excitation data

        Returns:
            torch.tensor: tensor of the scaled real and imaginary data for training
        """

        if type(data) == torch.Tensor:
            return data

        if self.resampled:
            bins = self.resampled_bins
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

    def test_train_split_(self, test_size=0.2, random_state=42, resampled=None, scaled=True, shuffle=True):
        """
        test_train_split_ Utility function that does the test train split for the neural network data

        Args:
            test_size (float, optional): fraction for the test size. Defaults to 0.2.
            random_state (int, optional): fixed seed for the random state. Defaults to 42.
            resampled (bool, optional): selects if should use resampled data. Defaults to True.
            scaled (bool, optional): selects if you should use the scaled data. Defaults to True.
            shuffle (bool, optional): selects if the data should be shuffled. Defaults to True.

        Returns:
            torch.tensor: X_train, X_test, y_train, y_test
        """

        # gets the neural network data
        x_data, y_data = self.NN_data(resampled, scaled)

        # does the test train split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=shuffle)

        # prints the extraction state
        if self.verbose:
            self.extraction_state

        return self.X_train, self.X_test, self.y_train, self.y_test

    class Raw_Data_Scaler():
        """
        Raw_Data_Scaler class that defines the scaler for band excitation data

        """

        def __init__(self, raw_data):
            """
            __init__ Initialization function

            Args:
                raw_data (np.array): raw band excitation data to scale
            """

            self.raw_data = raw_data

            # conduct the fit on initialization
            self.fit()

        @staticmethod
        def complex_data_converter(data):
            """
            complex_data_converter converter that converts the dataset to complex

            Args:
                data (np.array): band excitation data to convert

            Returns:
                np.array: band excitation data as a complex number
            """
            if BE_Dataset.is_complex(data):
                return data
            else:
                return BE_Dataset.to_complex(data)

        def fit(self):
            """
            fit function to fit the scaler
            """

            # gets the raw data
            data = self.raw_data
            data = self.complex_data_converter(data)

            # extracts the real and imaginary components
            real = np.real(data)
            imag = np.imag(data)

            # does a global scaler on the data
            self.real_scaler = global_scaler()
            self.imag_scaler = global_scaler()

            # computes global scaler on the real and imaginary parts
            self.real_scaler.fit(real)
            self.imag_scaler.fit(imag)

        def transform(self, data):
            """
            transform Function to transform the data

            Args:
                data (np.array): band excitation data

            Returns:
                np.array: scaled band excitation data_
            """

            # converts the data to a complex number
            data = self.complex_data_converter(data)

            # extracts the real and imaginary components
            real = np.real(data)
            imag = np.imag(data)

            # computes the transform
            real = self.real_scaler.transform(real)
            imag = self.imag_scaler.transform(imag)

            # returns the complex number
            return real + 1j*imag

        def inverse_transform(self, data):
            """
            inverse_transform Computes the inverse transform

            Args:
                data (np.array): band excitation data

            Returns:
                np.array: unscaled band excitation data
            """

            # converts the data to complex
            data = self.complex_data_converter(data)

            # extracts the real and imaginary componets
            real = np.real(data)
            imag = np.imag(data)

            # computes the inverse transform
            real = self.real_scaler.inverse_transform(real)
            imag = self.imag_scaler.inverse_transform(imag)

            return real + 1j*imag

    def get_loop_path(self):
        """
        get_loop_path gets the path where the hysteresis loops are located

        Returns:
            str: string pointing to the path where the hysteresis loops are located
        """

        if self.noise == 0 or self.noise is None:
            prefix = 'Raw_Data'
            return f"/{prefix}-SHO_Fit_000/Fit-Loop_Fit_000"
        else:
            prefix = f"Noisy_Data_{self.noise}"
            return f"/Noisy_Data_{self.noise}_SHO_Fit/Noisy_Data_{self.noise}-SHO_Fit_000/Guess-Loop_Fit_000"

    @static_state_decorator
    def get_hysteresis(self,
                       noise=None,
                       plotting_values=False,
                       output_shape=None,
                       scaled=None,
                       loop_interpolated=None,
                       measurement_state=None,
                       ):
        """
        get_hysteresis function to get the hysteresis loops

        Args:
            noise (int, optional): sets the noise value. Defaults to None.
            plotting_values (bool, optional): sets if you get the data shaped for computation or plotting. Defaults to False.
            output_shape (str, optional): sets the shape of the output. Defaults to None.
            scaled (any, optional): selects if the output is scaled or unscaled. Defaults to None.
            loop_interpolated (any, optional): sets if you should get the interpolated loops. Defaults to None.
            measurement_state (any, optional): sets the measurement state. Defaults to None.

        Returns:
            np.array: output hysteresis data, bias vector for the hystersis loop
        """

        # todo: can replace this to make this much nicer to get the data. Too many random transforms

        if measurement_state is not None:
            self.measurement_state = measurement_state

        with h5py.File(self.file, "r+") as h5_f:

            # sets the noise value
            if noise is None:
                self.noise = noise

            # sets the output shape
            if output_shape is not None:
                self.output_shape = output_shape

            # selects if the scaled data is returned
            if scaled is not None:
                self.scaled = scaled

            # selects if interpolated hysteresis loops are returned
            if loop_interpolated is not None:
                self.loop_interpolated = loop_interpolated

            # gets the path where the hysteresis loops are located
            h5_path = self.get_loop_path()

            # gets the projected loops
            h5_projected_loops = h5_f[h5_path + '/Projected_Loops']

            # Prepare some variables for plotting loops fits and guesses
            # Plot the Loop Guess and Fit Results
            proj_nd, _ = reshape_to_n_dims(h5_projected_loops)

            spec_ind = get_auxiliary_datasets(h5_projected_loops,
                                              aux_dset_name='Spectroscopic_Indices')[-1]
            spec_values = get_auxiliary_datasets(h5_projected_loops,
                                                 aux_dset_name='Spectroscopic_Values')[-1]
            pos_ind = get_auxiliary_datasets(h5_projected_loops,
                                             aux_dset_name='Position_Indices')[-1]

            pos_nd, _ = reshape_to_n_dims(pos_ind, h5_pos=pos_ind)
            pos_dims = list(pos_nd.shape[:pos_ind.shape[1]])

            # reshape the vdc_vec into DC_step by Loop
            spec_nd, _ = reshape_to_n_dims(spec_values, h5_spec=spec_ind)
            loop_spec_dims = np.array(spec_nd.shape[1:])
            loop_spec_labels = get_attr(spec_values, 'labels')

            spec_step_dim_ind = np.where(loop_spec_labels == 'DC_Offset')[0][0]

            # Also reshape the projected loops to Positions-DC_Step-Loop
            final_loop_shape = pos_dims + \
                [loop_spec_dims[spec_step_dim_ind]] + [-1]
            proj_nd2 = np.moveaxis(
                proj_nd, spec_step_dim_ind + len(pos_dims), len(pos_dims))
            proj_nd_3 = np.reshape(proj_nd2, final_loop_shape)

            # Get the bias vector:
            spec_nd2 = np.moveaxis(
                spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)
            bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims):])

            if plotting_values:
                proj_nd_3, bias_vec = self.roll_hysteresis(bias_vec, proj_nd_3)

            hysteresis_data = np.transpose(proj_nd_3, (1, 0, 3, 2))

            # interpolates the data
            if self.loop_interpolated:
                hysteresis_data = clean_interpolate(hysteresis_data)

            # transforms the data with the scaler if necessary.
            if self.scaled:
                hysteresis_data = self.hystersis_scaler.transform(
                    hysteresis_data)

            # sets the data to the correct output shape
            if self.output_shape == "index":
                hysteresis_data = proj_nd_3.reshape(
                    self.num_cycles*self.num_pix, self.voltage_steps//self.num_cycles)
            elif self.output_shape == "pixels":
                pass

            hysteresis_data = self.hysteresis_measurement_state(
                hysteresis_data)

        # output shape (x,y, cycle, voltage_steps)
        return hysteresis_data, np.swapaxes(np.atleast_2d(self.get_voltage), 0, 1).astype(np.float64) # bias_vec

    def get_bias_vector(self, plotting_values=True):

        # TODO: could look at get_hysteresis to simplify code

        with h5py.File(self.file, "r+") as h5_f:

            # gets the path where the hysteresis loops are located
            h5_path = self.get_loop_path()

            # gets the projected loops
            h5_projected_loops = h5_f[h5_path + '/Projected_Loops']

            spec_ind = get_auxiliary_datasets(h5_projected_loops,
                                              aux_dset_name='Spectroscopic_Indices')[-1]
            pos_ind = get_auxiliary_datasets(h5_projected_loops,
                                             aux_dset_name='Position_Indices')[-1]
            spec_values = get_auxiliary_datasets(h5_projected_loops,
                                                 aux_dset_name='Spectroscopic_Values')[-1]

            pos_nd, _ = reshape_to_n_dims(pos_ind, h5_pos=pos_ind)
            pos_dims = list(pos_nd.shape[:pos_ind.shape[1]])

            pos_dims = list(pos_nd.shape[:pos_ind.shape[1]])

            # reshape the vdc_vec into DC_step by Loop
            spec_nd, _ = reshape_to_n_dims(spec_values, h5_spec=spec_ind)
            loop_spec_labels = get_attr(spec_values, 'labels')
            spec_step_dim_ind = np.where(loop_spec_labels == 'DC_Offset')[0][0]

            loop_spec_dims = np.array(spec_nd.shape[1:])

            # Also reshape the projected loops to Positions-DC_Step-Loop
            final_loop_shape = pos_dims + \
                [loop_spec_dims[spec_step_dim_ind]] + [-1]

            # Get the bias vector:
            spec_nd2 = np.moveaxis(
                spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)

            bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims):])

            if plotting_values:
                bias_vec = self.roll_hysteresis(bias_vec)
                
            return bias_vec

    def hysteresis_measurement_state(self, hysteresis_data):
        """utility function to extract the measurement state from the hysteresis data

        Args:
            hysteresis_data (np.array): hysteresis data to extract the measurement state from

        Returns:
            np.array: hysterisis data with the measurement state extracted
        """

        if self.measurement_state == "all" or self.measurement_state is None:
            return hysteresis_data
        if self.measurement_state == "off":
            return hysteresis_data[:, :, hysteresis_data.shape[2]//2:hysteresis_data.shape[2], :]
        if self.measurement_state == "on":
            return hysteresis_data[:, :, 0:hysteresis_data.shape[2]//2, :]

    def roll_hysteresis(self, bias_vector, hysteresis=None,
                        shift=4):
        """
        roll_hysteresis function to shift the bias vector and the hysteresis loop by a quarter cycle. This is to compensate for the difference in how the data is stored.

        Args:
            hysteresis (np.array): array for the hysteresis loop
            bias_vector (np.array): array for the bias vector
            shift (int, optional): fraction to roll the hysteresis loop by. Defaults to 4.

        Returns:
            _type_: _description_
        """

        # TODO: long term this is likely the wrong way to do this, should get this from the USID file spectroscopic index

        # Shift the bias vector and the loops by a quarter cycle
        shift_ind = int(-1 * bias_vector.shape[0] / shift)
        bias_vector = np.roll(bias_vector, shift_ind, axis=0)
        if hysteresis is None:
            return bias_vector
        else:
            proj_nd_shifted = np.roll(hysteresis, shift_ind, axis=2)
            return proj_nd_shifted, bias_vector

    @property
    def BE_superposition_state(self):
        """
        BE_superposition_state get the BE superposition state

        Returns:
            str: gets the superposition state
        """
        with h5py.File(self.file, "r+") as h5_f:
            BE_superposition_state_ = h5_f["Measurement_000"].attrs['VS_measure_in_field_loops']
        return BE_superposition_state_

    def loop_shaper(self, data, shape="pixels"):
        """
        loop_shaper Tool to reshape the piezoelectric hystersis loops based on the desired shape

        Args:
            data (np.array): hysteresis loops to reshape
            shape (str, optional): pixel or index as a string to reshpae. Defaults to "pixels".

        Raises:
            ValueError: The data shape is not compatible with the number of rows and columns
            ValueError: The data shape is not compatible with the number of rows and columns

        Returns:
            np.array: reshaped piezoelectric hysteresis loops.
        """

        if shape == "pixels":
            try:
                return data.reshape(self.rows, self.cols, self.voltage_steps, self.num_cycles)
            except:
                raise ValueError(
                    "The data shape is not compatible with the number of rows and columns")
        if shape == "index":
            try:
                return data.reshape(self.num_pix_1d, self.voltage_steps, self.num_cycles)
            except:
                raise ValueError(
                    "The data shape is not compatible with the number of rows and columns")

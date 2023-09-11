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
                 noise=0,
                 cleaned=False,
                 basegroup='/Measurement_000/Channel_000',
                 SHO_fit_func_LSQF=SHO_fit_func_nn,
                 hysteresis_function=None,
                 loop_interpolated=False,
                 **kwargs):

        self.file = file_
        self.resampled = resampled
        self.scaled = scaled
        self.raw_format = raw_format
        self.fitter = fitter
        self.output_shape = output_shape
        self.measurement_state = measurement_state
        self.basegroup = basegroup
        self.hysteresis_function = hysteresis_function
        self.loop_interpolated = loop_interpolated
        self.tree = self.get_tree()
        self.cleaned = cleaned

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
        self.set_raw_data()

    def generate_noisy_data_records(self, noise_levels,
                                    basegroup='/Measurement_000/Channel_000',
                                    verbose=False,
                                    noise_STD=None):

        if noise_STD is None:
            noise_STD = np.std(self.get_original_data)

        if verbose:
            print(f"The STD of the data is: {noise_STD}")

        with h5py.File(self.file, "r+") as h5_f:

            for noise_level in noise_levels:

                if verbose:
                    print(f"Adding noise level {noise_level}")

                noise_level_ = noise_STD * noise_level

                noise_real = np.random.uniform(-1*noise_level_,
                                               noise_level_, (3600, 63360))
                noise_imag = np.random.uniform(-1*noise_level_,
                                               noise_level_, (3600, 63360))
                noise = noise_real+noise_imag*1.0j
                data = self.get_original_data + noise

                h5_main = usid.hdf_utils.find_dataset(h5_f, "Raw_Data")[0]

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

        if in_list(self.tree, "*SHO_Fit*"):
            self.SHO_preprocessing()
        else:
            Warning("No SHO fit found")

        if in_list(self.tree, "*Fit-Loop_Fit*"):
            self.loop_fit_preprocessing()

    def loop_fit_preprocessing(self):

        hysteresis, bias = self.get_hysteresis(
            plotting_values=True, output_shape="index")

        cleaned_hysteresis = clean_interpolate(hysteresis)
        self.hystersis_scaler = global_scaler()
        self.hystersis_scaler.fit_transform(cleaned_hysteresis)

    def SHO_preprocessing(self):
        # extract the raw data and reshapes is
        self.set_raw_data()

        # resamples the data if necessary
        self.set_raw_data_resampler()

        # computes the scalar on the raw data
        self.raw_data_scaler = self.Raw_Data_Scaler(self.raw_data())

        self.set_SHO_LSQF()
        self.SHO_Scaler()

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

        self.set_attributes(**default_state_)

    def get_tree(self):

        with h5py.File(self.file, "r+") as h5_f:
            return get_tree(h5_f)

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
                   dataset="Raw_Data",
                   h5_sho_targ_grp=None):
        """Function that computes the SHO fit results

        Args:
            force (bool, optional): forces the SHO results to be computed from scratch. Defaults to False.
            max_cores (int, optional): number of processor cores to use. Defaults to -1.
            max_mem (_type_, optional): maximum ram to use. Defaults to 1024*8.
        """

        # something strange with the fitter
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
        if self.noise == 0:
            return "Raw_Data_SHO_Fit"
        else:
            return f"Noisy_Data_{self.noise}"

    def LSQF_Loop_Fit(self, main_dataset='/Raw_Data_SHO_Fit/Raw_Data-SHO_Fit_000/Fit', h5_target_group=None,
                      max_cores=None):

        with h5py.File(self.file, "r+") as h5_file:

            expt_type = sidpy.hdf.hdf_utils.get_attr(h5_file, 'data_type')

            h5_meas_grp = usid.hdf_utils.find_dataset(
                h5_file, self.measure_group())

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

            loop_fitter = belib.analysis.BELoopFitter(main_dataset,
                                                      expt_type, vs_mode, vs_cycle_frac,
                                                      h5_target_group=h5_target_group,
                                                      cores=max_cores,
                                                      verbose=False)
            loop_fitter.set_up_guess()
            h5_loop_guess = loop_fitter.do_guess(override=False)

            # Calling explicitly here since Fitter won't do it automatically
            h5_guess_loop_parms = loop_fitter.extract_loop_parameters(
                h5_loop_guess)
            loop_fitter.set_up_fit()
            h5_loop_fit = loop_fitter.do_fit(override=False)
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
    def LSQF_hysteresis_params(self, output_shape=None, scaled=None):
        """
        LSQF_hysteresis_params Gets the LSQF hysteresis parameters

        Args:
            output_shape (str, optional): pixel or list. Defaults to None.
            scaled (bool, optional): selects if to scale the data. Defaults to None.

        Returns:
            np.array: hysteresis loop parameters from LSQF
        """

        if output_shape is not None:
            self.output_shape = output_shape

        if scaled is not None:
            self.scaled = scaled

        with h5py.File(self.file, "r+") as h5_f:
            data = h5_f[f"/{self.dataset}_SHO_Fit/{self.dataset}-SHO_Fit_000/Fit-Loop_Fit_000/Fit"][:]
            data = data.reshape(self.num_rows, self.num_cols, self.num_cycles)
            data = np.array([data['a_0'], data['a_1'], data['a_2'], data['a_3'], data['a_4'],
                            data['b_0'], data['b_1'], data['b_2'], data['b_3']]).transpose((1, 2, 3, 0))

            if self.scaled:
                # TODO: add the scaling here
                Warning("Scaling not implemented yet")
                pass

            if self.output_shape == "index":
                data = data.reshape(
                    self.num_pix, self.num_cycles, data.shape[-1])

            return data

    @static_state_decorator
    def SHO_Scaler(self,
                   noise=0):

        # set the noise and the dataset
        self.noise = noise

        self.SHO_scaler = StandardScaler()
        data = self.SHO_LSQF().reshape(-1, 4)

        self.SHO_scaler.fit(data)

        # sets the phase not to scale
        self.SHO_scaler.mean_[3] = 0
        self.SHO_scaler.var_[3] = 1
        self.SHO_scaler.scale_[3] = 1

    def SHO_LSQF(self, pixel=None, voltage_step=None):
        with h5py.File(self.file, "r+") as h5_f:

            dataset_ = self.SHO_LSQF_data[f"{self.dataset}-SHO_Fit_000"].copy()

            if pixel is not None and voltage_step is not None:
                return self.get_voltage_state(dataset_[[pixel], :, :])[:, [voltage_step], :]
            elif pixel is not None:
                return self.get_voltage_state(dataset_[[pixel], :, :])
            else:
                return self.get_voltage_state(dataset_[:])

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
    def to_complex(data, axis=None):
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

        return np.take(data, 0, axis=axis) + 1j * np.take(data, 1, axis=axis)

    def set_SHO_LSQF(self,
                     scaler="Raw_Data-SHO_Fit_000",
                     save_loc='SHO_LSQF'):
        """Utility function to convert the SHO fit results to an array

        Args:
            SHO_LSQF (h5 Dataset): Location of the fit results in an h5 file

        Returns:
            np.array: SHO fit results
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

                self.SHO_LSQF_data[name] = data_.reshape(
                    self.num_pix, self.voltage_steps, 5)[:, :, :-1]

    @staticmethod
    def shift_phase(phase, shift_=None):

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
        """Resampled real part of the complex data resampled"""
        if pixel is not None and voltage_step is not None:
            return self.resampled_data[self.dataset][[pixel], :, :][:, [voltage_step], :]
        else:
            with h5py.File(self.file, "r+") as h5_f:
                return self.resampled_data[self.dataset][:]

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

    def state_num_voltage_steps(self):

        if self.measurement_state == 'all':
            voltage_step = self.voltage_steps
        else:
            voltage_step = int(self.voltage_steps/2)

        return voltage_step

    def SHO_fit_results(self,
                        state=None,
                        model=None,
                        phase_shift=None,
                        X_data=None):

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

    def get_cycle(self, data, axis=0,  **kwargs):
        data = np.array_split(data, self.num_cycles, axis=axis, **kwargs)
        data = data[self.cycle - 1]
        return data

    def get_measurement_cycle(self, data, cycle=None, axis=1):
        if cycle is not None:
            self.cycle = cycle
        data = self.get_voltage_state(data)
        return self.get_cycle(data, axis=axis)

    @static_state_decorator
    def get_raw_data_from_LSQF_SHO(self, model, index=None):

        self.set_attributes(**model)

        self.scaled = False

        params_shifted = self.SHO_fit_results()

        exec(f"self.{model['fitter']}_phase_shift=0")

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

        return pred_data, params

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if kwargs.get("noise"):
            self.noise = kwargs.get("noise")

    @static_state_decorator
    def raw_spectra(self,
                    pixel=None,
                    voltage_step=None,
                    fit_results=None,
                    type_="numpy",
                    frequency=False,
                    noise=None,
                    state=None):
        """Raw spectra"""

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

        if length == self.num_bins:
            x = self.frequency_bin
        elif length == self.resampled_bins:
            x = resample(self.frequency_bin,
                         self.resampled_bins)
        else:
            raise ValueError(
                "original data must be the same length as the frequency bins or the resampled frequency bins")
        return x

    def shaper(self, data, pixel=None, voltage_steps=None, length=None):

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
            voltage_steps = int(self.voltage_steps.copy())

            if self.measurement_state in ["on", "off"]:
                voltage_steps /= 2
                voltage_steps = int(voltage_steps)

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
                for data in self.raw_datasets:
                    resampled_ = self.resampler(
                        self.raw_data_reshaped[data].reshape(-1, self.num_bins), axis=2)
                    self.resampled_data[data] = resampled_.reshape(
                        self.num_pix, self.voltage_steps, self.resampled_bins)
            else:
                self.resampled_data = self.raw_data_reshaped

            if kwargs.get("basepath"):
                self.data_writer(kwargs.get("basepath"), save_loc, resampled_)

    def resampler(self, data, axis=2):
        """Resample the data to a given number of bins"""
        with h5py.File(self.file, "r+") as h5_f:
            try:
                return resample(data.reshape(self.num_pix, -1, self.num_bins),
                                self.resampled_bins, axis=axis)
            except ValueError:
                print("Resampling failed, check that the number of bins is defined")

    @property
    def extraction_state(self):
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
    def NN_data(self, resampled=True, scaled=True):
        print(self.extraction_state)
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

    def get_loop_path(self):
        if self.noise == 0 or self.noise is None:
            prefix = 'Raw_Data'
            return f"/{prefix}_SHO_Fit/{prefix}-SHO_Fit_000/Fit-Loop_Fit_000"
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
                       ):

        with h5py.File(self.file, "r+") as h5_f:

            if noise is None:
                self.noise = noise

            if output_shape is not None:
                self.output_shape = output_shape

            if scaled is not None:
                self.scaled = scaled

            if loop_interpolated is not None:
                self.loop_interpolated = loop_interpolated

            h5_path = self.get_loop_path()

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
                proj_nd_3, bias_vec = self.roll_hysteresis(proj_nd_3, bias_vec)

            hysteresis_data = np.transpose(proj_nd_3, (1, 0, 3, 2))

            if self.loop_interpolated:
                hysteresis_data = clean_interpolate(hysteresis_data)

            if self.scaled:
                hysteresis_data = self.hystersis_scaler.transform(
                    hysteresis_data)

            if self.output_shape == "index":
                hysteresis_data = proj_nd_3.reshape(
                    self.num_cycles*self.num_pix, self.voltage_steps//self.num_cycles)
            elif self.output_shape == "pixels":
                pass

        # output shape (x,y, cycle, voltage_steps)
        return hysteresis_data, bias_vec

    def roll_hysteresis(self, hysteresis, bias_vector, shift=4):

        # Shift the bias vector and the loops by a quarter cycle
        shift_ind = int(-1 * bias_vector.shape[0] / 4)
        proj_nd_shifted = np.roll(hysteresis, shift_ind, axis=2)
        bias_vector = np.roll(bias_vector, shift_ind, axis=0)

        return proj_nd_shifted, bias_vector

    @property
    def BE_superposition_state(self):
        with h5py.File(self.file, "r+") as h5_f:
            BE_superposition_state_ = h5_f["Measurement_000"].attrs['VS_measure_in_field_loops']
        return BE_superposition_state_

    def loop_shaper(self, data, shape="pixels"):

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

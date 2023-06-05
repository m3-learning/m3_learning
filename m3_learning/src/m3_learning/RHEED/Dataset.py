import h5py
import numpy as np
import matplotlib.pyplot as plt
from m3_learning.viz.layout import imagemap, layout_fig, labelfigs
from m3_learning.RHEED.Viz import Viz
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


def compress_gaussian_params_H5(file_in, str=None, compression='gzip', compression_opts=9):
    """
    Compresses Gaussian parameters in an HDF5 file.

    Args:
        file_in (str): Path to the input HDF5 file.
        str (str, optional): String to append to the output file name. Defaults to None.
        compression (str, optional): Compression algorithm to use. Defaults to 'gzip'.
        compression_opts (int, optional): Compression level. Defaults to 9.
    """

    if str is None:
        out_name = file_in[:-3] + '_compressed.h5'
    else:
        out_name = file_in[:-3] + str

    with h5py.File(f"{file_in}", 'r') as f_old:
        print(f_old.keys())
        with h5py.File(out_name, 'w') as f_new:
            for ds in f_old.keys():
                f_new.create_group(ds)
                for spot in f_old[ds].keys():
                    f_new[ds].create_group(spot)
                    for metric in f_old[ds][spot].keys():
                        data = f_old[ds][spot][metric][:]
                        dset = f_new[ds][spot].create_dataset(
                            metric, data=data, compression=compression, compression_opts=compression_opts)
                        
def compress_RHEED_spot_H5(file_in, str=None, compression='gzip', compression_opts=9):
    """
    Compresses RHEED spots in an HDF5 file.

    Args:
        file_in (str): Path to the input HDF5 file.
        str (str, optional): String to append to the output file name. Defaults to None.
        compression (str, optional): Compression algorithm to use. Defaults to 'gzip'.
        compression_opts (int, optional): Compression level. Defaults to 9.
    """
    
    if str is None:
        out_name = file_in[:-3] + '_compressed.h5'
    else:
        out_name = file_in[:-3] + str
    
    with h5py.File(file_in, 'r') as f_old:
        print(f_old.keys())
        with h5py.File(out_name, 'w') as f_new:
            for growth in f_old.keys():
                print(growth)
                data = f_old[growth][:]
                dset = f_new.create_dataset(
                    growth, data=data, compression=compression, compression_opts=compression_opts)


class RHEED_spot_Dataset:
    """A class representing a dataset of RHEED spots.

    Attributes:
        path (str): The path to the dataset.
        sample_name (str): The name of the sample.
        verbose (bool): Whether to enable verbose mode or not.

    Methods:
        data_info: Prints information about the dataset.
        growth_dataset: Retrieves the RHEED spot data for a specific growth.
        viz_RHEED_spot: Visualizes a specific RHEED spot.

    Properties:
        sample_name: Getter and setter for the sample name.

    """
    def __init__(self, path, sample_name, verbose=False):
        """
        Initializes a new instance of the RHEED_spot_Dataset class.

        Args:
            path (str): The path to the dataset.
            sample_name (str): The name of the sample.
            verbose (bool, optional): Whether to enable verbose mode or not. Defaults to False.
        """
        self.path = path
        self._sample_name = sample_name

    @property
    def data_info(self):
        """
        Prints information about the dataset.

        This method reads the dataset file and prints the growth names along with the size of their data arrays.
        """
        ...
        with h5py.File(self.path, mode='r') as h5:
            for g, data in h5.items():
                try:
                    print(f"Growth: {g}, Size of data: f{data.shape}")
                except:
                    print(f"Growth: {g}")

    def growth_dataset(self, growth, index = None):
        """
        Retrieves the RHEED spot data for a specific growth.

        Args:
            growth (str): The name of the growth.
            index (int, optional): The index of the data array to retrieve. Defaults to None.

        Returns:
            numpy.ndarray: The RHEED spot data as a numpy array.

        Raises:
            ValueError: If the index is out of range.

        """
        with h5py.File(self.path, mode='r') as h5:
            if index is None:
                return np.array(h5[growth])
            else:
                if index<0 or index>h5[growth].shape[0]:
                    raise ValueError('Index out of range')
                else:
                    return np.array(h5[growth][index])
                                
    def viz_RHEED_spot(self, growth, index, figsize=(2, 2), clim=None, filename = None, printing=None, **kwargs):
        """
        Visualizes a specific RHEED spot.

        Args:
            growth (str): The name of the growth.
            index (int): The index of the data array to visualize.
            figsize (tuple, optional): The size of the figure. Defaults to (2, 2).
            clim (tuple, optional): The color limit for the plot. Defaults to None.
            filename (str or bool, optional): The filename to save the plot. If True, a default filename will be used. Defaults to None.
            printing: A printing object used for saving the figure. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the printing object.

        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # fig, axes = layout_fig(1, figsize=figsize)

        data = self.growth_dataset(growth, index)
        # imagemap(axes[0], data, clim=clim, divider_=True)
        # customized version of imagemap
        im = ax.imshow(data)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        cbar = fig.colorbar(im, ticks=[data.min(), data.max(), np.mean([data.min(), data.max()])], cax=cax, format="%.2e")
        ax.set_yticklabels("")
        ax.set_xticklabels("")
        ax.set_yticks([])
        ax.set_xticks([])
        labelfigs(ax, 0)
        if filename is True: 
            filename = f"RHEED_{self.sample_name}_{growth}_{index}"

        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)

        print(f'\033[1mFig.\033[0m a: RHEED spot image for {growth} at index {index}.')
        plt.show()

    @property
    def sample_name(self):
        """
        Getter for the sample name.

        Returns:
            str: The name of the sample.
        """
        return self._sample_name
    
    @sample_name.setter
    def sample_name(self, sample_name):
        """
        Setter for the sample name.

        Returns:
            sample_name (str): The name of the sample.
        """

        self._sample_name = sample_name


class RHEED_parameter_dataset():
    """A class representing a dataset of RHEED spots with associated parameters.

    Attributes:
        path (str): The path to the dataset.
        camera_freq (float): The camera frequency.
        sample_name (str): The name of the sample.
        verbose (bool): Whether to enable verbose mode or not.

    Methods:
        data_info: Prints information about the dataset.
        growth_dataset: Retrieves the RHEED spot parameter data for a specific growth, spot, and metric.
        load_curve: Loads a parameter curve for a specific growth, spot, and metric.
        load_multiple_curves: Loads multiple parameter curves for a list of growths, spot, and metric.
        viz_RHEED_parameter: Visualizes the RHEED spot parameter for a specific growth, spot, and index.
        viz_RHEED_parameter_trend: Visualizes the parameter trends for multiple growths, spot, and metrics.

    Properties:
        camera_freq: Getter and setter for the camera frequency.
        sample_name: Getter and setter for the sample name.

    """

    def __init__(self, path, camera_freq, sample_name, verbose=False):
        """
        Initializes a new instance of the RHEED_parameter_dataset class.

        Args:
            path (str): The path to the dataset.
            camera_freq (float): The camera frequency.
            sample_name (str): The name of the sample.
            verbose (bool, optional): Whether to enable verbose mode or not. Defaults to False.
        """
        self.path = path
        self._camera_freq = camera_freq
        self._sample_name = sample_name
        
    @property
    def data_info(self):
        """
        Prints information about the dataset.

        This method reads the dataset file and prints the growth names, spot names, and the size of their associated data arrays.
        """
        with h5py.File(self.path, mode='r') as h5:
            for g in h5.keys():
                print(f"Growth: {g}:")
                for s in h5[g].keys():
                    print(f"--spot: {s}:")
                    for k in h5[g][s].keys():
                        try:
                            print(f"----{k}:, Size of data: {h5[g][s][k].shape}")
                            print(f"----{k}:, Size of data: {h5[g][s][k].shape}")
                        except:
                            print(f"----metric: {k}")

                            print(f"----metric: {k}")


    def growth_dataset(self, growth, spot, metric, index = None):
        """
        Retrieves the RHEED spot parameter data for a specific growth, spot, and metric.

        Args:
            growth (str): The name of the growth.
            spot (str): The name of the spot.
            metric (str): The name of the metric. Options: "raw_image", "reconstructed_image", "img_sum", "img_max", "img_mean", 
        "img_rec_sum", "img_rec_max", "img_rec_mean", "height", "x", "y", "width_x", "width_y".
            index (int, optional): The index of the data array to retrieve. Defaults to None.

        Returns:
            numpy.ndarray: The RHEED spot parameter data as a numpy array.

        Raises:
            ValueError: If the index is out of range.

        Options for metric: 

        """

        with h5py.File(self.path, mode='r') as h5:
            
            if index is None:
                return np.array(h5[growth][spot][metric])
            else:
                if index<0 or index>h5[growth][spot][metric].shape[0]:
                    raise ValueError('Index out of range')
                else:
                    return np.array(h5[growth][spot][metric][index])

    def load_curve(self, growth, spot, metric, x_start):
        """
        Loads a parameter curve for a specific growth, spot, and metric.

        Args:
            growth (str): The name of the growth.
            spot (str): The name of the spot.
            metric (str): The name of the metric.
            x_start (float): The starting x value for the curve.

        Returns:
            tuple: A tuple containing the x and y values of the parameter curve.

        """
        with h5py.File(self.path, mode='r') as h5_para:
            y = np.array(h5_para[growth][spot][metric])
        x = np.linspace(x_start, x_start+len(y)-1, len(y))/self.camera_freq
        return x, y

    def load_multiple_curves(self, growth_list, spot, metric, x_start=0, head_tail=(100, 100), interval=200):
        """
        Loads multiple parameter curves for a list of growths, spot, and metric.

        Args:
            growth_list (list): The list of growth names.
            spot (str): The name of the spot.
            metric (str): The name of the metric.
            x_start (float, optional): The starting x value for the first curve. Defaults to 0.
            head_tail (tuple, optional): The number of elements to remove from the head and tail of each curve. Defaults to (100, 100).
            interval (int, optional): The interval between curves. Defaults to 200.

        Returns:
            tuple: A tuple containing the concatenated x and y values of the parameter curves.

        """
        x_all, y_all = [], []
        
        for growth in growth_list:
            x, y = self.load_curve(growth, spot, metric, x_start)
            x = x[head_tail[0]:-head_tail[1]]
            y = y[head_tail[0]:-head_tail[1]]
            x_start = x_start+len(y)+interval
            x_all.append(x)
            y_all.append(y)

        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
        return x_all, y_all

    def viz_RHEED_parameter(self, growth, spot, index, figsize=None, filename=None, printing=None, **kwargs):
        """
        Visualizes the RHEED spot parameter for a specific growth, spot, and index.

        Args:
            growth (str): The name of the growth.
            spot (str): The name of the spot.
            index (int): The index of the data array to visualize.
            figsize (tuple, optional): The size of the figure. Defaults to None.
            filename (str, optional): The filename to save the plot. Defaults to None.
            printing: A printing object used for saving the figure. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the printing object.

        """
        if figsize is None:
            figsize = (1.25*3, 1.25*1)
        # "img_mean", "img_rec_sum", "img_rec_max", "img_rec_mean", "height", "x", "y", "width_x", "width_y".
        img = self.growth_dataset(growth, spot, 'raw_image', index)
        img_rec = self.growth_dataset(growth, spot, 'reconstructed_image', index)
        img_sum = self.growth_dataset(growth, spot, 'img_sum', index)
        img_max = self.growth_dataset(growth, spot, 'img_max', index)
        img_mean = self.growth_dataset(growth, spot, 'img_mean', index)
        img_rec_sum = self.growth_dataset(growth, spot, 'img_rec_sum', index)
        img_rec_max = self.growth_dataset(growth, spot, 'img_rec_max', index)
        img_rec_mean = self.growth_dataset(growth, spot, 'img_rec_mean', index)
        height = self.growth_dataset(growth, spot, 'height', index)
        x = self.growth_dataset(growth, spot, 'x', index)
        y = self.growth_dataset(growth, spot, 'y', index)
        width_x = self.growth_dataset(growth, spot, 'width_x', index)
        width_y = self.growth_dataset(growth, spot, 'width_y', index)

        sample_list = [img, img_rec, img_rec-img]

        clim = (img.min(), img.max())
        fig, axes = layout_fig(3, 3, figsize=(1.25*3, 1.25*1))
        for i, ax in enumerate(axes):
            if ax == axes[-1]:
                imagemap(ax, sample_list[i], divider_=False, clim=clim, colorbars=True, **kwargs)
            else:
                imagemap(ax, sample_list[i], divider_=False, clim=clim, colorbars=False, **kwargs)
            labelfigs(ax, i)

        if filename is True: 
            filename = f"RHEED_{self.sample_name}_{growth}_{spot}_{index}_img,img_rec,differerce"
                
        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)
        plt.show()
        print(f'\033[1mFig.\033[0m a: RHEED spot image, b: reconstructed RHEED spot image, c: difference between original and reconstructed image for {growth} at index {index}.')
        #print first 2 digits of each parameter
        print(f'img_sum={img_sum:.2f}, img_max={img_max:.2f}, img_mean={img_mean:.2f}')
        print(f'img_rec_sum={img_rec_sum:.2f}, img_rec_max={img_rec_max:.2f}, img_rec_mean={img_rec_mean:.2f}')
        print(f'height={height:.2f}, x={x:.2f}, y={y:.2f}, width_x={width_x:.2f}, width_y_max={width_y:.2f}')
        

    def viz_RHEED_parameter_trend(self, growth_list, spot, metric_list=None, head_tail=(100, 100), interval=0, filename = None, printing=None, **kwargs):
        """
        Visualizes the parameter trends for multiple growths, spot, and metrics.

        Args:
            growth_list (list): The list of growth names.
            spot (str): The name of the spot.
            metric_list (list, optional): The list of metrics to visualize. Defaults to None.
            filename (str, optional): The filename to save the plot. Defaults to None.
            printing: A printing object used for saving the figure. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the printing object.

        """
        if metric_list is None:
            metric_list = ['img_sum', 'img_rec_sum', 'x', 'y', 'width_x', 'width_y']
        
        if len(metric_list) == 1:
            fig, ax = plt.subplots(len(metric_list), 1, figsize = (6, 2))
            axes = [ax]
        else:
            fig, axes = plt.subplots(len(metric_list), 1, figsize = (6, 1.5*len(metric_list)))
        for i, (ax, metric) in enumerate(zip(axes, metric_list)):
            x_curve, y_curve = self.load_multiple_curves(growth_list, spot=spot, metric=metric, head_tail=head_tail, interval=interval) #**kwargs)
            ax.scatter(x_curve, y_curve, color='k', s=1)
            if i < len(metric_list)-1:
                Viz.set_labels(ax, ylabel=f'{metric} (a.u.)', yaxis_style='sci')
                ax.set_xticklabels(['' for tick in ax.get_xticks()])
            else:
                Viz.set_labels(ax, xlabel='Time (s)', ylabel=f'{metric} (a.u.)', yaxis_style='sci')
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))  # Adjust the power limits as needed
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_x(-0.05)
            labelfigs(ax, i, label=metric, loc='tl', style='b', size=8, inset_fraction=(0.08, 0.03))

        fig.subplots_adjust(hspace=0)

        if filename: 
            filename = f"RHEED_{self.sample_name}_{spot}_metrics"
                
        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)
        plt.show()
        print(f'Gaussian fitted parameters in time: \033[1mFig.\033[0m a: sum of original image, b: sum of reconstructed image, c: spot center in spot x coordinate, d: spot center in y coordinate, e: spot width in x coordinate, f: spot width in y coordinate.')


    @property
    def camera_freq(self):
        """
        Getter for the camera frequency.

        Returns:
            float: The camera frequency.
        """
        return self._camera_freq

    @camera_freq.setter
    def camera_freq(self, camera_freq):
        """
        Setter for the camera frequency.

        Args:
            camera_freq (float): The new camera frequency.
        """
        self._camera_freq = camera_freq

    @property
    def sample_name(self):
        """
        Getter for the sample name.

        Returns:
            str: The name of the sample.
        """
        return self._sample_name
    
    @sample_name.setter
    def sample_name(self, sample_name):
        """
        Setter for the sample name.

        Args:
            sample_name (str): The new name for the sample.
        """
        self._sample_name = sample_name
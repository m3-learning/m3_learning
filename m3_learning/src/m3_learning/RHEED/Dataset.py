import h5py
import numpy as np
import matplotlib.pyplot as plt
from m3_learning.viz.layout import imagemap, layout_fig, labelfigs
from m3_learning.RHEED.Viz import Viz
import matplotlib.pyplot as plt
from m3_learning.viz.layout import imagemap, layout_fig, labelfigs
from m3_learning.RHEED.Viz import Viz

class RHEED_spot_Dataset:
    def __init__(self, path, sample_name, verbose=False):
        self.path = path
        self._sample_name = sample_name

    @property
    def data_info(self):
        with h5py.File(self.path, mode='r') as h5:
            for g, data in h5.items():
                try:
                    print(f"Growth: {g}, Size of data: f{data.shape}")
                except:
                    print(f"Growth: {g}")

    def growth_dataset(self, growth, index = None):
        with h5py.File(self.path, mode='r') as h5:
            if index is None:
                return np.array(h5[growth])
            else:
                if index<0 or index>h5[growth].shape[0]:
                    raise ValueError('Index out of range')
                else:
                    return np.array(h5[growth][index])
                                
    def viz_RHEED_spot(self, growth, index, figsize=None, clim=None, filename = None, printing=None, **kwargs):
        if figsize is None: figsize = (1.5, 1.5)
        fig, axes = layout_fig(1, figsize=figsize)
#         fig, ax = plt.subplots(figsize = figsize)
        data = self.growth_dataset(growth, index)
        imagemap(axes[0], data, clim=clim, divider_=True)

        if filename is True: 
            filename = f"RHEED_{self.sample_name}_{growth}_{index}"

        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)
        plt.show()

    def viz_RHEED_spot(self, growth, index, figsize=None, clim=None, filename = None, printing=None, **kwargs):

        if figsize is None: figsize = (1.5, 1.5)
        fig, axes = layout_fig(1, figsize=figsize)
#         fig, ax = plt.subplots(figsize = figsize)
        data = self.growth_dataset(growth, index)
        imagemap(axes[0], data, clim=clim, divider_=True)

        if filename is True: 
            filename = f"RHEED_{self.sample_name}_{growth}_{index}"

        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)
        plt.show()

    @property
    def sample_name(self):
        return self._sample_name
    
    @sample_name.setter
    def sample_name(self, sample_name):
        self._sample_name = sample_name


class RHEED_parameter_dataset():

    def __init__(self, path, camera_freq, sample_name, verbose=False):
        self.path = path
        self._camera_freq = camera_freq
        self._sample_name = sample_name
        
    @property
    @property
    def data_info(self):
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
    def growth_dataset(self, growth, spot, metric, index = None):
        '''
        Options for metric: "raw_image", "reconstructed_image", "img_sum", "img_max", "img_mean", 
        "img_rec_sum", "img_rec_max", "img_rec_mean", "height", "x", "y", "width_x", "width_y".
        Options for metric: "raw_image", "reconstructed_image", "img_sum", "img_max", "img_mean", 
        "img_rec_sum", "img_rec_max", "img_rec_mean", "height", "x", "y", "width_x", "width_y".
        '''
        with h5py.File(self.path, mode='r') as h5:
            if index is None:
                return np.array(h5[growth][spot][metric])
                return np.array(h5[growth][spot][metric])
            else:
                if index<0 or index>h5[growth][spot][metric].shape[0]:
                if index<0 or index>h5[growth][spot][metric].shape[0]:
                    raise ValueError('Index out of range')
                else:
                    return np.array(h5[growth][spot][metric][index])

    def load_curve(self, growth, spot, metric, x_start):
        with h5py.File(self.path, mode='r') as h5_para:
            y = np.array(h5_para[growth][spot][metric])
        x = np.linspace(x_start, x_start+len(y)-1, len(y))/self.camera_freq
        return x, y

    def load_multiple_curves(self, growth_list, spot, metric, x_start=0, head_tail=(100, 100), interval=200):
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

    def viz_RHEED_parameter(self, growth, spot, index, figsize=None, filename = None, printing=None, **kwargs):
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

        print(f'img_sum:{img_sum:.2f}, img_max:{img_max:.2f}, img_mean:{img_mean:.2f}')
        print(f'img_rec_sum:{img_rec_sum:.2f}, img_rec_max:{img_rec_max:.2f}, img_rec_mean:{img_rec_mean:.2f}')
        print(f'height:{height:.2f}, x:{x:.2f}, y:{y:.2f}, width_x:{width_x:.2f}, width_y_max:{width_y:.2f}')

        sample_list = [img, img_rec, img_rec-img]
        print('a: raw_image', 'b: reconstructed_image', 'c: difference')
        fig, axes = layout_fig(3, 3, figsize=figsize)
        for i, ax in enumerate(axes):
            ax.imshow(sample_list[i])
            labelfigs(ax, i)
            ax.axis("off")

        if filename is True: 
            filename = f"RHEED_{self.sample_name}_{growth}_{spot}_{index}_img,img_rec,differerce"
                
        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)
        plt.show()

    def viz_RHEED_parameter_trend(self, growth_list, spot, metric_list=None, figsize=(8, 2), filename = None, printing=None, **kwargs):
    
        if metric_list is None:
            metric_list = ['img_sum', 'img_max', 'img_mean', 'img_rec_sum', 'img_rec_max', 'img_rec_mean', 'height', 'x', 'y', 'width_x', 'width_y']
        for i, metric in enumerate(metric_list):
            fig, ax = plt.subplots(figsize=figsize)
            x_curve, y_curve = self.load_multiple_curves(growth_list, spot=spot, metric=metric, **kwargs)
            ax.scatter(x_curve, y_curve, color='k', s=1)
            Viz.set_labels(ax, xlabel='Time (s)', ylabel=f'{metric} (a.u.)')
            
            if filename: 
                filename = f"RHEED_{self.sample_name}_{spot}_{metric}"
                    
            # prints the figure
            if printing is not None and filename is not None:
                printing.savefig(fig, filename, **kwargs)
            plt.show()

    @property
    def camera_freq(self):
        return self._camera_freq

    @camera_freq.setter
    def camera_freq(self, camera_freq):
        self._camera_freq = camera_freq

    @property
    def sample_name(self):
        return self._sample_name
    
    @sample_name.setter
    def sample_name(self, sample_name):
        self._sample_name = sample_name
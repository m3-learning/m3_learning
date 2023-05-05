import os
import h5py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, sosfilt, freqz
from scipy import optimize
from joblib import Parallel, delayed
import sys
from m3_learning.viz.layout import layout_fig, labelfigs
from m3_learning.RHEED.Viz import Viz


def NormalizeData(data, lb=0, ub=1):
    return (data - lb) / (ub - lb)

def show_fft_frequency(amplitude, samplingFrequency, ranges=None):

    # Frequency domain representation
    fourierTransform = np.fft.fft(
        amplitude)/len(amplitude)           # Normalize amplitude
    fourierTransform = fourierTransform[range(
        int(len(amplitude)/2))]  # Exclude sampling frequency

    tpCount = len(amplitude)
    values = np.arange(int(tpCount/2))
    timePeriod = tpCount/samplingFrequency

    frequencies = values/timePeriod
    fourierTransform[abs(fourierTransform) > 1] = 0
    if ranges:
        frequencies_ = frequencies[frequencies > ranges[0]]
        fourierTransform_ = fourierTransform[frequencies > ranges[0]]

        frequencies_range = frequencies_[frequencies_ < ranges[1]-ranges[0]]
        fourierTransform_range = fourierTransform_[
            frequencies_ < ranges[1]-ranges[0]]
    else:
        frequencies_range = frequencies
        fourierTransform_range = fourierTransform

    plt.figure(figsize=(15, 4))
    plt.plot(frequencies_range, abs(fourierTransform_range))
    plt.show()
    return frequencies_range, abs(fourierTransform_range)


def butter_filter(data, method, filter_type, cutoff, samplingFrequency, order):
    nyq = 0.5 * samplingFrequency

    if type(cutoff) == tuple:
        cutoff = list(cutoff)

    if type(cutoff) == list:
        cutoff[0] = cutoff[0] / nyq
        cutoff[1] = cutoff[1] / nyq
    else:
        cutoff = cutoff / nyq

    if method == 'ba':
        b, a = butter(order, cutoff, btype=filter_type,
                      analog=False, output='ba')
        y = lfilter(b, a, data)
    if method == 'sos':
        sos = butter(order, cutoff, btype=filter_type,
                     analog=False, output='sos')
        y = sosfilt(sos, data)
    return y


def process_pass_filter(sound, filter_type, method, cutoff, order, frame_range, samplingFrequency=100):
    sig = np.copy(sound)
    t = np.arange(0, len(sig))
    ranges = None

    filtered = butter_filter(sig, method, filter_type,
                             cutoff, samplingFrequency, order)

    show_fft_frequency(sig, samplingFrequency, ranges)
    show_fft_frequency(filtered, samplingFrequency, ranges)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(t[frame_range[0]:frame_range[1]],
             sig[frame_range[0]:frame_range[1]], marker='v')

    ax2.plot(t[frame_range[0]:frame_range[1]],
             filtered[frame_range[0]:frame_range[1]], marker='v')
    ax2.set_xlabel('Frame')
    plt.tight_layout()
    plt.show()
    return filtered


def show_metrics(data, ranges, plot_ranges):

    len_img = 16
    img_per_row = 8
    fig, ax = plt.subplots(len_img//img_per_row+1*int(len_img % img_per_row > 0),
                           img_per_row, figsize=(16, 2*len_img//img_per_row+1))
    for i in range(len_img):
        ax[i//img_per_row, i % img_per_row].title.set_text(i)
        if ranges:
            ax[i//img_per_row, i %
                img_per_row].imshow(data[i][ranges[0]:ranges[1], ranges[2]:ranges[3]])
        else:
            ax[i//img_per_row, i % img_per_row].imshow(data[i])

    plt.show()

    sum_list, max_list, min_list, mean_list, std_list = [], [], [], [], []
    for i in range(len(data)):
        if ranges:
            img = data[i][ranges[0]:ranges[1], ranges[2]:ranges[3]]
        else:
            img = data[i]
        sum_list.append(np.sum(img))
        max_list.append(np.max(img))
        min_list.append(np.min(img))
        mean_list.append(np.mean(img))
        std_list.append(np.std(img))

    fig, ax = plt.subplots(3, 2, figsize=(15, 12))

    if ranges:
        h = ax[0, 0].plot(sum_list[plot_ranges[0]:plot_ranges[1]])
        ax[0, 0].title.set_text('sum_list')

        h = ax[0, 1].plot(max_list[plot_ranges[0]:plot_ranges[1]])
        ax[0, 1].title.set_text('max_list')

        h = ax[1, 0].plot(min_list[plot_ranges[0]:plot_ranges[1]])
        ax[1, 0].title.set_text('min_list')

        h = ax[1, 1].plot(mean_list[plot_ranges[0]:plot_ranges[1]])
        ax[1, 1].title.set_text('mean_list')

        h = ax[2, 0].plot(std_list[plot_ranges[0]:plot_ranges[1]])
        ax[2, 0].title.set_text('std_list')

    else:
        h = ax[0, 0].plot(sum_list)
        ax[0, 0].title.set_text('sum_list')

        h = ax[0, 1].plot(max_list)
        ax[0, 1].title.set_text('max_list')

        h = ax[1, 0].plot(min_list)
        ax[1, 0].title.set_text('min_list')

        h = ax[1, 1].plot(mean_list)
        ax[1, 1].title.set_text('mean_list')

        h = ax[2, 0].plot(std_list)
        ax[2, 0].title.set_text('std_list')

    plt.show()
    return sum_list, max_list, min_list, mean_list, std_list


# add referece for Josh's repository
class Gaussian():
    def __init__(self):
        self.a = 0

    def gaussian(self, height, center_x, center_y, width_x, width_y, rotation):
        """Returns a gaussian function with the given parameters"""

        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

        def rotgauss(x, y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width_x)**2 +
                  ((center_y-yp)/width_y)**2)/2.)
            return g
        return rotgauss

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y, 0.0

    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        def errorfunction(p): return np.ravel(
            self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p

    def recreate_gaussian(self, image):
        para = self.fitgaussian(image)
        y = np.linspace(0, image.shape[0], image.shape[0])
        x = np.linspace(0, image.shape[1], image.shape[1])
        x, y = np.meshgrid(x, y)
        return self.gaussian(*para)(y, x), para


class RHEED_image_processer:
    def __init__(self, spot_ds, crop_dict, fit_function):
        '''
        spots_names = ['spot_1', 'spot_2', 'spot_3']
        '''
        self.spot_ds = spot_ds
        self.crop_dict = crop_dict
        self.fit_function = fit_function

    def write_h5_file(self, parameters_file_path, growth_list, replace=False, num_workers=1):
        '''
        parameters: img_sum, img_max, img_mean, img_rec_sum, img_rec_max, img_rec_mean, height, x, y, width_x, width_y
        '''
        spots_names = list(self.crop_dict.keys())

        if os.path.isfile(parameters_file_path):
            print('h5 file exist.')
            if replace:
                os.remove(parameters_file_path)
                print('Replace with new file.')
        with h5py.File(parameters_file_path, mode='a') as h5_para:
            for growth in growth_list:
                h5_growth = h5_para.create_group(growth)
                for spot in spots_names:
                    inputs = self.normalize_inputs(self.spot_ds.growth_dataset(growth), spot)
                    results = self.fit_batch(inputs, num_workers)

                    img_all = np.array([res[0] for res in results])
                    img_rec_all = np.array([res[1] for res in results])
                    parameters = np.array([res[2] for res in results])

                    h5_spot = h5_growth.create_group(spot)
                    h5_spot.create_dataset('raw_image', data=img_all)
                    h5_spot.create_dataset('reconstructed_image', data=img_rec_all)
                    h5_spot.create_dataset('img_sum', data=parameters[:, 0])
                    h5_spot.create_dataset('img_max', data=parameters[:, 1])
                    h5_spot.create_dataset('img_mean', data=parameters[:, 2])
                    h5_spot.create_dataset('img_rec_sum', data=parameters[:, 3])
                    h5_spot.create_dataset('img_rec_max', data=parameters[:, 4])
                    h5_spot.create_dataset('img_rec_mean', data=parameters[:, 5])
                    h5_spot.create_dataset('height', data=parameters[:, 6])
                    h5_spot.create_dataset('x', data=parameters[:, 7])
                    h5_spot.create_dataset('y', data=parameters[:, 8])
                    h5_spot.create_dataset('width_x', data=parameters[:, 9])
                    h5_spot.create_dataset('width_y', data=parameters[:, 10])

    def normalize_inputs(self, data, spot):
        crop = self.crop_dict[spot]
        if len(data.shape) == 2:
            inputs = NormalizeData(data[crop[0]:crop[1], crop[2]:crop[3]])
        elif len(data.shape) == 3:
            inputs = NormalizeData(
                np.array(data[:, crop[0]:crop[1], crop[2]:crop[3]]))
        return inputs

    def fit_batch(self, inputs, num_workers):
        if num_workers > 1:
            tasks = [delayed(self.fit)(img) for img in inputs]
            results = Parallel(n_jobs=num_workers)(tasks)
        else:
            results = [self.fit(img) for img in inputs]
        return results

    def fit(self, img):
        # para: height, x, y, width_x, width_y, 0.0
        img_rec, para = self.fit_function(img)
        img_sum, img_max, img_mean = np.sum(img), np.max(img), np.mean(img)
        img_rec_sum, img_rec_max, img_rec_mean = np.sum(img_rec), np.max(img_rec), np.mean(img_rec)
        parameters = [img_sum, img_max, img_mean,
                      img_rec_sum, img_rec_max, img_rec_mean, *para]
        return img, img_rec, parameters

    def visualize(self, growth, spot, frame):

        img = self.spot_ds.growth_dataset(growth, frame)
        img = self.normalize_inputs(img, spot)
        img, img_rec, parameters = self.fit(img)
        print(
            #print first 2 digits of each parameter
            f'img_sum:{parameters[0]:.2f}, img_max:{parameters[1]:.2f}, img_mean:{parameters[2]:.2f}')
        print(
            f'img_rec_sum:{parameters[3]:.2f}, img_rec_max:{parameters[4]:.2f}, img_rec_mean:{parameters[5]:.2f}')
        print(
            f'height:{parameters[6]:.2f}, x:{parameters[7]:.2f}, y:{parameters[8]:.2f}, width_x:{parameters[9]:.2f}, width_y_max:{parameters[10]:.2f}')
        
        sample_list = [img, img_rec, img_rec-img]
        print('a: raw_image', 'b: reconstructed_image', 'c: difference')
        fig, axes = layout_fig(3, 3, figsize=(1.25*3, 1.25*1))
        for i, ax in enumerate(axes):
            ax.imshow(sample_list[i])
            labelfigs(ax, i)

        plt.show()
        print('a: original, b: reconstructed image, c: difference')
        return img, img_rec, parameters

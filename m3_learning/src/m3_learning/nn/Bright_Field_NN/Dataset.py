import numpy as np
import hyperspy.api as hs
import os
import h5py
from skimage.color import rgb2gray
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import skimage
from tqdm import tqdm
import imageio
from pprint import pprint
import time
import math
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections.abc import Iterable

# Pycroscopy
import sidpy
import pyNSID
# print('sidpy version: ', sidpy.__version__)
import pycroscopy as px
from pycroscopy.image import ImageWindowing
import dask.array as da

class Bright_Field_Dataset:

    def __init__(self, datapath, combined_name='', verbose=False):
        """Initialization function

        Args:
            path (str): folder where data is saved. This dataset is only compatible 
                with temperature dependent data, in folders named 'Ramp_up' and 'Ramp_down'
            combined_name (str, optional): prefix for the h5 file to save the cropped images 
                and windowed training data. Resulting dataset is called <combined_name>_combined.h5
                and saved in the datapath folder. Defaults to ''.
            verbose (bool, optional): whether to print completion messages. Defaults to False.
        """
        self.path =datapath
        self.combined_name = combined_name
        self.verbose = verbose
        if len(self.combined_name) > 0:
            self.combined_h5_path = self.path +"/" + self.combined_name + '_combined.h5'
        else:         
            self.combined_h5_path = self.path +"/" + self.combined_name + 'combined.h5'

        # make list of file names for original bright field images
        self.temps = self.get_temp_paths()
        self.t_len = len(self.temps)
        
        if self.verbose:
            print('number of images: ', self.t_len)

    def temp(self, e):
        '''
        Gets temperature from file in format temp.png and determines ramp up or down
        
        Args:
            e (str): directory to extract temperature from
        '''
        if len(e) < 8:
            return '-'+e[:-4].zfill(3)
        else:
            return '-0.png'


    def list_imgs(self):
        """Creates list of images labels in format f'{Ramp_Up/Down} {temperature}'

        Returns:
            List: List of input image labels
        """        
        l = [file.split('/')[-2] + ' ' + file.split('/')[-1][:-4] for file in self.temps]
        # pprint(l)
        return l


    def get_raw_img(self, **kwargs):
        """Gets raw image from files

        Args:
            path_index (int): which number image in to use, when images are ordered according to time taken
            
            OR
            
            state (str): "Ramp_Up" or "Ramp_Down"
            temperature (int/str): temperature image taken at
            
        Returns:
            numpy.ndarray: raw image
        """        
        if len(kwargs.keys())==1:
            s = self.get_temp_paths()[kwargs['path_index']]
            
        if len(kwargs.keys())==2:
            s = f"{self.path}/{kwargs['state']}/{kwargs['temperature']}.png"
            
        im = rgb2gray(imageio.imread(s))
        return im


    def get_index(self):
        pass


    def get_temp_paths(self):
        """gets list of paths to files of all temperatures

        Returns:
            List: paths to files of all temperatures, sorted by order they were taken
        """
        up = os.listdir(f'{self.path}/Ramp_Up')
        up.sort(key=self.temp)
        up = list(map(lambda x: f'{self.path}/Ramp_Up/'+x, up))

        down = os.listdir(f'{self.path}/Ramp_Down')
        down.sort(key=self.temp, reverse=True)
        down = list(map(lambda x: f'{self.path}/Ramp_Down/'+x, down))

        temps = up+down
        return temps


    def get_combined_h5_info(self):
        """Recursively find all keys in h5 file containing raw images, 
            filtered images, and windowed data. Prints them out in pretty format
        """          

        def flatten(xs):
            for x in xs:
                if isinstance(x, list) and not isinstance(x, (str, bytes)):
                    yield from flatten(x)
                else:
                    yield x
                    
        def allkeys(obj,layer=0):
            "Recursively find all keys in an h5py.Group."
            keys = [(obj.name,layer)]
            if isinstance(obj, h5py.Group):
                for key, value in obj.items():
                    if isinstance(value, h5py.Group):
                        keys.append(allkeys(value,layer=layer+1)[0])
                    else:
                        keys.append((value.name,layer+1))
            return keys,layer+1
                    
        with h5py.File(self.combined_h5_path,'a') as h:
            keys = allkeys(h)
            # return keys
            key_layers = flatten(keys)
            
            for f in key_layers:
                if isinstance(f,tuple):
                    print(f[1]*'\t',end='')
                    print(h[f[0]])


    def get_shape(self,dataset_name):
        """get the shape of the 

        Args:
            dataset_name (string): path to h5 dataset

        Returns:
            sh (tuple): = shape of h5 dataset
        """       
        with h5py.File(self.combined_h5_path,'a') as h:
            sh = h[dataset_name].shape
        return sh


    def get_window_index(self,t,a,b,
                         dataset_key):
        """gets the scan from temperature idx t and point a,b, on the sample.

        Args:
            t (int): temperature index
            a (int): window index in x direction
            b (int): window index in y direction
            windows_group (str, optional): name of the h5 group where window datasets are written. Defaults to 'windows'.
            dset_name (str, optional): name of the raw h5 dataset. Defaults to 'windows_data'.
            logset_name (str, optional): name of the log+threshold h5 dataset. Defaults to 'windows_logdata'.
            filtered (bool, optional): _description_. Defaults to True.

        Returns:
            idx (integer): index of the window in the flattened input datset
            bbox (list): bounding corners [x1,x2,y1,y2]
        """        
        with h5py.File(self.combined_h5_path,'a') as h:
            # try:
            grp = h[dataset_key]
            orig = h['All_filtered']
            t_,ox,oy = orig.shape
            _,a_,b_,_,_ = grp.attrs['orig_shape']
            x_ = int(np.ceil(grp.attrs['window_size_x']/2))
            y_ = int(np.ceil(grp.attrs['window_size_y']/2))
            x = x_ + a*grp.attrs['window_step_x']
            y = x_ + b*grp.attrs['window_step_y']

            idx = t*a_*b_+b*b_+a
            bbox = [x-x_, x+x_, y-y_, y+y_]
            return idx, bbox


    def open_combined_h5(self):
        """Opens the combined h5 file. Not recommended because it file is 
        easily corrupted if not closed immediately after use

        Returns:
            H5 File: combined h5 file
        """        
        if len(self.combined_name) > 0:
            combined = self.path +"/" + self.combined_name + '_combined.h5'
        else:         
            combined = self.path +"/" + self.combined_name + 'combined.h5'

        return h5py.File(combined,'a')


    def NormalizeData(self,data):
        '''min-max normalization
        
        Args:
            data (array like)
        '''
        if np.max(data)==np.min(data): return data
        else: return (data - np.min(data)) / (np.max(data) - np.min(data))


    def write_h5(self, x0, y0, width):
        """Creates h5_file if it doesn't already exist. 
        Crops raw data intoi square and create a dataset with and without Gaussian filtering.

        Args:
            x0 (int): lower x bound
            y0 (int): lower y bound
            width (int): width/height of cropped image
        """
        t_len = len(self.temps)
        
        with h5py.File(self.combined_h5_path,'a') as h:
            if 'All' in h.keys():
                del h['All']
            
            h_write = h.create_dataset('All', (t_len, width, width), dtype='f4')

            if 'All_filtered' in h.keys():
                del h['All_filtered']
            
            h_writef = h.create_dataset(
                    'All_filtered', (t_len, width, width), dtype='f4')

            for i, temp_ in enumerate(tqdm(self.temps)):
                im = rgb2gray(imageio.imread(temp_))
                im1 = im[x0:x0+width, y0:y0+width]  # crop

                h_write[i] = im1

                img_blur = ndimage.gaussian_filter(im1, 20)
                im1 = im1-img_blur
                # self.scaler = MinMaxScaler() # v2
                # im1 = self.scaler.fit_transform(im1)
                im1 = self.NormalizeData(im1)

                h_writef[i] = im1


    def write_windows(self,windows_group = 'windows',
                      dset_name = 'windows_data',
                      logset_name = 'windows_logdata',
                      target_size=128,
                      filter_threshold=5,
                      overwrite = False,
                      window_parms={'fft_mode': 'abs',
                                    'interpol_factor': 2.3,
                                    'mode': 'fft',
                                    'window_size_x': 128,
                                    'window_size_y': 128,
                                    'window_step_x': 32,
                                    'window_step_y': 32,
                                    'zoom_factor': 2,
                                    'filter': 'hamming'}
                ):
        """Generate and take tiled windows with applied parameters. 
        Writes a raw version and version with logarithm and thresholding applied to h5 file

        Args:
            name (str): name of sample (ex 'Annealed','Oxygen')
            window_params (dict, optional): parameters to give pycroscopy windowing function. Defaults to {}.
            windows_group (str, optional): name of the h5 group where window datasets are written. Defaults to 'windows'.
            dset_name (str, optional): name of the raw h5 dataset. Defaults to 'windows_data'.
            logset_name (str, optional): name of the log+threshold h5 dataset. Defaults to 'windows_logdata'.
            target_size (int, optional): how many pixels are in the final log dataset. Defaults to 128.
            filter_threshold (int, optional): cutoff for log dataset. Defaults to 5.
            overwrite (bool): whether or not to delete existing window file
            window_params (dict): parameters to give to pycroscopy's windowing function
                Default: {'fft_mode': 'abs',
                                    'interpol_factor': 2.3,
                                    'mode': 'fft',
                                    'window_size_x': 128,
                                    'window_size_y': 128,
                                    'window_step_x': 32,
                                    'window_step_y': 32,
                                    'zoom_factor': 2,
                                    'filter': 'hamming'}

        Returns:
        returns ImageWindowing object. From pycroscopy.image package
        """     
        # window_parms['interpol_factor'] = target_size/window_parms['window_size_x']*window_parms['zoom_factor'] # v2
        iw = ImageWindowing(window_parms)
        with h5py.File(self.combined_h5_path,'a') as h:
            # set windows group
            if windows_group not in h.keys(): 
                h_windows=h.create_group(windows_group)
            h_windows=h[windows_group]
            
            # for image taken at each temperature
            for i,img_path in enumerate(tqdm(self.list_imgs())):
                # create compatible dataset
                im_dataset = sidpy.Dataset.from_array( 
                    h['All_filtered'][i]
                        )
                # make windows (Takes longest)
                windows = iw.MakeWindows(im_dataset)

                # create temporary dataset to store windows data and get shape
                if f'filler' in h[windows_group]: del h[windows_group][f'filler']
                filler = h[windows_group].create_group(f'filler')
                pyNSID.hdf_io.write_nsid_dataset(windows, filler, main_data_name="windows");
                a,b,x,y = h[windows_group]['filler']['windows']['windows'].shape
                data = np.abs(h[windows_group]['filler']['windows']['windows'][:].reshape(-1,x,y))

                # if overwriting, delete existing dataset in combined h5 file
                if overwrite: 
                    if dset_name in h[windows_group].keys(): del h[windows_group][dset_name]
                    d_windows=h[windows_group].create_dataset(dset_name,
                                                            shape=(self.t_len*a*b,x,y),
                                                            dtype='f4')
                    if logset_name in h[windows_group].keys(): del h[windows_group][logset_name]
                    logdata= h[windows_group].create_dataset(logset_name,
                                                            shape=(self.t_len*a*b,target_size,target_size),
                                                            dtype='f4')
                # otherwise create dataset h5 combined dataset if it doesn't already exist.
                else:
                    if dset_name not in h[windows_group].keys(): 
                        d_windows=h[windows_group].create_dataset(dset_name,
                                                                shape=(self.t_len*a*b,x,y),
                                                                dtype='f4')
                    if logset_name not in h[windows_group].keys(): 
                        logdata= h[windows_group].create_dataset(logset_name,
                                                                shape=(self.t_len*a*b,target_size,target_size),
                                                                dtype='f4')            
                d_windows=h[windows_group][dset_name]
                logdata=h[windows_group][logset_name]
                
                # write raw fft into dataset
                d_windows[i*a*b:(i+1)*a*b] = data

                # scale and write into processed dataset
                # data -= data.min(axis=(0)) # v2
                data = data.reshape(-1,x,y)
                data = resize(data,(a*b,target_size,target_size)) # v1
                # print(a,b,x,y,target_size)
                data = np.log(data+1)
                if filter_threshold:
                    data[data>filter_threshold]=filter_threshold
                # scaler = MinMaxScaler() # v2
                # data = scaler.fit_transform(data.reshape(a*b,-1)) # v2
                logdata[i*a*b:(i+1)*a*b] = data

            if self.verbose: # print shape and 
                print('shape: ',a,b,target_size,target_size)
                pprint(window_parms)
            
            # write metadata for windows
            d_windows.attrs['orig_shape'] = (self.t_len,a,b,x,y)
            logdata.attrs['orig_shape'] = (self.t_len,a,b,target_size,target_size)
            logdata.attrs['filter_threshold'] = filter_threshold
            for key,val in window_parms.items(): # write metadata
                d_windows.attrs[key]=val
                logdata.attrs[key]=val
            self.shape = logdata.shape

        return iw

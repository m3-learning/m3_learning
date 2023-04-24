import numpy as np
import hyperspy.api as hs
import os
import h5py
from skimage.color import rgb2gray
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import imageio
from pprint import pprint

# Pycroscopy
import sidpy
import pyNSID
print('sidpy version: ', sidpy.__version__)
import pycroscopy as px
from pycroscopy.image import ImageWindowing
import dask.array as da

class Bright_Field_Dataset:

    def __init__(self, path, filename, verbose=False):

        self.path = path
        self.filename = filename
        self.verbose = verbose

        # make list of file names for original bright field images
        self.temps = self.get_temps()
        self.t_len = len(self.temps)
        
        if self.verbose:
            print('number of images: ', self.t_len)

    def temp(self, e):
        '''
        Gets temperature from file in format temp.png and determines ramp up or down
        '''
        if len(e) < 8:
            return '-'+e[:-4].zfill(3)
        else:
            return '-0.png'

    def list_imgs(self):
        l = [file.split('/')[-2] + ' ' + file.split('/')[-1][:-4] for file in self.temps]
        pprint(l)

    def get_img(self, state, temperature):
        """_summary_

        Args:
            state (_type_): "Up" or "Down"
            temperature (_type_): _description_
        """        
        s = f"{self.path}/Ramp_{state}/{temperature}.png"
        im = rgb2gray(imageio.imread(s))
        return im


    def get_temps(self):

        up = os.listdir(f'{self.path}/Ramp_Up')
        up.sort(key=self.temp)
        up = list(map(lambda x: f'{self.path}/Ramp_Up/'+x, up))

        down = os.listdir(f'{self.path}/Ramp_Down')
        down.sort(key=self.temp, reverse=True)
        down = list(map(lambda x: f'{self.path}/Ramp_Down/'+x, down))

        temps = up+down
        return temps

    def write_h5(self, c1, c2, step, name=""):
        """_summary_

        Args:
            c1 (_type_): _description_
            c2 (_type_): _description_
            step (_type_): _description_
            name (str, optional): _description_. Defaults to "".
        """
        
        if len(name) > 0:
            name += '_'

        combined = self.path +"/" + name + 'preprocessed.h5'

        # put the name of the file you would like to use
        h = h5py.File(combined, 'a')
        t_len = len(self.temps)

        if 'All' in h.keys():
            del h['All']
        
        h_write = h.create_dataset('All', (t_len, step, step), dtype='f4')

        if 'All_filtered' in h.keys():
            del h['All_filtered']
        
        h_writef = h.create_dataset(
                'All_filtered', (t_len, step, step), dtype='f4')

        for i, temp_ in enumerate(tqdm(self.temps)):
            im = rgb2gray(imageio.imread(temp_))
            im1 = im[c1:c1+step, c2:c2+step]  # crop

            h_write[i] = im1

            img_blur = ndimage.gaussian_filter(im1, 20)
            im1 = im1-img_blur
            self.scaler = MinMaxScaler()
            im1 = self.scaler.fit_transform(im1)

            h_writef[i] = im1

        h.close()

    def write_windows(self,name,window_params={},
                      windows_group = 'windows',
                      dset_name = 'windows_data',
                      logset_name = 'windows_logdata',
                      target_size=128,
                      filter_threshold=5
                  ):
        
        iw = ImageWindowing(window_params)
        
        combined = self.path +"/" + name + 'preprocessed.h5'
        h = h5py.File(combined,'a')
        if windows_group not in h.keys(): 
            h_windows=h.create_group(windows_group)
        h_windows=h[windows_group]

        for key,val in window_params.items(): # write metadata
            h_windows.attrs[key]=val
        
        for i,img_path in enumerate(tqdm(self.list_imgs())):
            im_data = sidpy.Dataset.from_array( self.get_img(img_path) )
            windows_group.MakeWindows(im_data)

            if f'filler' in h[windows_group]: 
                del h[windows_group][f'filler']
            filler = h[windows_group].create_group(f'filler')
            pyNSID.hdf_io.write_nsid_dataset(windows_group, filler, main_data_name="windows");

            a,b,x,y = h[windows_group]['filler']['windows']['windows'].shape
            data = h[windows_group]['filler']['windows']['windows'][:].reshape(-1,x,y)

            if dset_name not in h[windows_group].keys(): 
                d_windows=h[windows_group].create_dataset(dset_name,shape=(self.t_len*a*b,x,y))
            d_windows=h[windows_group][dset_name]
            d_windows[i*a*b:(i+1)*a*b] = data

            if logset_name not in h[windows_group].keys(): 
                logdata= h[windows_group].create_dataset(logset_name,
                                                         shape=(self.t_len*a*b,1,target_size,target_size),
                                                         dtype='f4')
            logdata=h[windows_group][logset_name]
            data = data.reshape(-1,1,x,y)
            data = skimage.transform.resize(data,(a*b,1,target_size,target_size))
            data = np.log(data+1)
            data[data>filter_threshold]=filter_threshold
            logdata[i*a*b:(i+1)*a*b] = data

            if self.verbose: # print shape and 
                print('shape: ',a,b,target_size,target_size)
        
        for key,val in window_params.items(): # write metadata
            d_windows.attrs[key]=val
            logdata.attrs[key]=val

        d_windows.attrs['orig_shape'] = (self.t_len,a,b,x,y)
        logdata.attrs['orig_shape'] = (self.t_len,a,b,target_size,target_size)
        logdata.attrs['filter_threshold'] = filter_threshold

        h.close()


# class Bright_Field_Dataset:
#     """Class for the STEM dataset.
#     """

#     def __init__(self, data_path):
#         """Initialization of the class.

#         Args:
#             data_path (string): path where the hyperspy file is located
#         """

#     self.data_path = data_path


    # @property
    # def log_data(self):
    #     return self._log_data

    # @log_data.setter
    # def log_data(self, log_data):
    #     # add 1 to avoid log(0)
    #     self._log_data = np.log(log_data.data + 1)

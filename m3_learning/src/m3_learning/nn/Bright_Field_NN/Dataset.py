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

        '''
        Crop and filter raw images and insert them into h5 file. Overwrites any exisiting crops
        c1: starting pixel
        c2: ending pixel
        step: pixel size to crop to
        combined: combined file name to write to
        temps: list of paths to BF images
        '''
        
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

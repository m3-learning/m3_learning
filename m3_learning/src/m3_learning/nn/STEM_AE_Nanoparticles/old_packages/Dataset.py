import numpy as np
import hyperspy.api as hs
import h5py
from skimage.draw import disk
import dask.array as da
import pyNSID
import os
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import Normalizer,StandardScaler
from tqdm import tqdm as tqdm

class STEM_Dataset:
    """Class for the STEM dataset.
    """

    def __init__(self, save_path, data_path,**kwargs):
        """Initialization of the class.

        Args:
            data_path (string): path where the hyperspy file is located
        """
        self.save_path = save_path
        self.data_path = data_path
        self.h5_name = f'{save_path}/combined_data.h5'

        def meta_from_dict_recursive(h5file, path, dic):
            """
            write dict to h5py group attributes recursively
            """
            for key, item in dic.items():
                if not isinstance(item, dict):
                    h5file[path].attrs[key] = item
                else:
                    meta_from_dict_recursive(h5file, path, item)
        
        if not os.path.exists(self.h5_name): h = h5py.File(self.h5_name,'w')
        else: h = h5py.File(self.h5_name,'r+')
        
        if 'raw_data' not in h:
            self.stacked = ('*' in data_path)
            # loads the data
            s = hs.load(data_path,
                    lazy=True,
                    stack=self.stacked
                    )
            if self.stacked: self.metadata = s.metadata.as_dictionary()
            else: self.metadata=s.original_metadata.as_dictionary()

            h.create_dataset('raw_data',data=s.data,dtype=float)
            # meta_from_dict_recursive(h,'raw_data',self.metadata)

        t,a,b,x,y = h['raw_data'].shape

        if 'processed' not in h: 
            h.create_dataset('processed',data=np.log(h['raw_data'][:].reshape((-1,x,y)) + 1),dtype=float)

        self.data = h['raw_data']
        self.processed = h['processed']

        # h.close()

        # sets the log data
        # self.log_data = s

        # self.processed=self.log_data.copy()

        # with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        #     self.processed=self.log_data.reshape(-1,
        #                                       self.data.shape[-2],
        #                                       self.data.shape[-1])

    # @property
    # def log_data(self):
    #     return self._log_data

    # @log_data.setter
    # def log_data(self, log_data):
    #     # add 1 to avoid log(0)
    #     self._log_data = np.log(log_data.data + 1)

    def crop(self,bbox):
        (bx1,bx2,by1,by2)=bbox
        h = h5py.File(self.h5_name,'r+')
        del h['processed']
        h.create_dataset('processed',
                         data=np.log(h['raw_data'][bx1:bx2,by1,by2] + 1),
                         dtype=float)
        h.close()

    def subtract_background(self,**kwargs):
        h = h5py.File(self.h5_name,'r+')
        print('Background Subtraction')
        for i,img in enumerate(tqdm(h['processed'][:])):
            h['processed'][i] -= gaussian_filter(img,**kwargs)
        h.close()
    
    def apply_scaler(self):
        h = h5py.File(self.h5_name,'r+')
        t,a,b,x,y = h['raw_data'].shape
        data = h['processed'][:].T.reshape(x*y,-1)
        print('standard scaling')
        data = StandardScaler().fit_transform(data)
        print('normalizing 0-1')
        data -= data.min(axis=0)
        data /= data.max(axis=0)
        print('writing to h5')
        h['processed'][:] = data.reshape(y,x,-1).T
        h.close()
        
    def apply_mask(self,bbox=None,center=None,radius=None):
        """apply a mask in the shape of a circle. 
        Arguments can either include a square around the brightfield or the center and radius

        Args:
            square (tuple, optional): (x1,x2,y1,y2) of bounding box. Defaults to None.
            center (tuple, optional): (x,y) indices of center of mask. Defaults to None.
            radius (int, optional): radius of mask. Defaults to None.
        """        
        h = h5py.File(self.h5_name,'r+')
        if not isinstance(bbox,type(None)): 
            (bx1,bx2,by1,by2)=bbox
            center = int((bx1+bx2)/2),int((by1+by2)/2)
            radius = max(abs(bx2-center[0])+1,abs(by2-center[1])+1)
        elif isinstance(center,type(None)) and isinstance(radius,type(None)):
            print("Enter either bounding box, or center and radius")
            return
        rr,cc = disk(center,radius)
        mask = np.ones((h['processed'].shape[1],h['processed'].shape[2]))
        mask[cc,rr] = 0
        print('Masking')
        for i,data in enumerate(tqdm(h['processed'][:])):
            h['processed'][i]=data*mask+(-mask+1)*h['processed'][i].mean()
        h.close()

    def apply_threshold(self,thresh):
        h = h5py.File(self.h5_name,'r+')
        args = np.argwhere(h['processed']>thresh)
        h['processed'][args] = thresh
        h.close()

    # def preprocess(self,mask_center=False,crop=False,sub_bkg=False,thresh=False):
    #     # mask_center
    #     # try:
    #         (bx1,bx2,by1,by2)=mask_center
    #         center = int((bx1+bx2)/2),int((by1+by2)/2)
    #         r = max(abs(bx2-center[0])+1,abs(by2-center[1])+1)
    #         rr,cc = disk(center,r)

    #         # mask=np.zeros((self.data.shape[-2],self.data.shape[-1]))
    #         # mask[rr,cc] = 0
    #         # self.processed = da.map_blocks(lambda x: x[:,:,:, rr, cc] * 0.0, 
    #         #                                self.processed, 
    #         #                                dtype=self.processed.dtype,
    #         #                                chunks=self.processed.chunks)
    #         # self.processed=test
    #     #     for i,j in list(zip(rr,cc)):
    #     #             self.processed[:,:,:,i,j] = 0
    #     # # except: print('No mask')


import matplotlib.pyplot as plt
from m3_learning.viz.layout import imagemap, add_scalebar
from m3_learning.util.file_IO import make_folder
from m3_learning.viz.layout import layout_fig,subfigures,plot_into_graph
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import io
import PIL
from skimage import morphology
from skimage.filters import threshold_otsu
import h5py

from tqdm import tqdm
from os.path import join as pjoin
from m3_learning.viz.nn import embeddings as embeddings_

class Viz:

    def __init__(self,
                 dset,
                 channels=None,
                 color_map='viridis',
                 printer=None,
                 labelfigs_=False,
                 scalebar_=None,
                 ):
        """Initialization function
        
        Args:
            dset (Bright_Field_Dataset object): 
            channels (list): channels to view embeddings. Defaults to None
            color_map (str): matplotlib colormap to use. Defaults to 'viridis'
            printer (printer object): Defaults to None
            labelfigs_ (bool): Whether to label figures. Defaults to False
            scalebar_ (dict): How to format scalebar. Defaults to None
        """
        

        self.dset = dset
        self.printer = printer
        self.labelfigs_ = labelfigs_
        self.scalebar_ = scalebar_
        self.cmap = plt.get_cmap(color_map)
        self.channels = channels



    def view_raw(self, img_name):
        """View sand saves raw image taked from saved folder

        Args:
            img_name (list): [state: "Ramp_Up"/"Ramp_Down", temperature]
        """        
        data = self.dset.get_raw_img(state=img_name[0],temperature=img_name[1])
        fig,axs = plt.subplots(figsize=(1.25,1.25))

        imagemap(axs, data, divider_ = True)

        axs.set_box_aspect(1)

        if self.scalebar_ is not None:
            # adds a scalebar to the figure
            add_scalebar(axs, self.scalebar_)

        if self.printer is not None:
              self.printer.savefig(fig,
                                     f'{img_name[0]}_{img_name[1]}_raw', 
                                     tight_layout=False,showfig=True)
        plt.show()
        plt.close('all')


    def view_window(self,img_name,x,y,
                    dataset_key,
                    scalebar_=True,
                    windows_group = 'windows',
                    dset_name = 'windows_data',
                    logset_name = 'windows_logdata',
                    view_windows = 'windows_logdata'):              
        '''
        Plot the filtered image, transform, and window
        
        Args:
            img_name (list): the condition and temperature of the image we want to see.
                In format ['Ramp_Up' or 'Ramp_Down', 'temperature']
            x (int): x tile
            y (int): y tile
            condition (str): 'Ramp_up' or 'Ramp_Down'
            cropped_scalebar (tuple) = Defaults to None,
            windows_group (str) = Default 'windows',
            dset_name (str) Default = 'windows_data',
            logset_name (str) = Default  'windows_logdata',
            view_windows (str) = Default 'windows_logdata'
        '''

        with h5py.File(self.dset.combined_h5_path,'a') as h:
            image = h['All_filtered']
            raw = self.dset.get_raw_img(state=img_name[0],temperature=img_name[1])

            for t,temp in enumerate(self.dset.temps):
                if f'{img_name[0]}/{img_name[1]}' in temp: break
            # print(t)

            # Make grid
            fig = plt.figure(figsize=(4,8/3))
            gs = fig.add_gridspec(2, 3)
            axs = []
            axs.append( fig.add_subplot(gs[:,0:2]) ) # large subplot (2 rows, 2 columns)
            axs.append( fig.add_subplot(gs[0,2]) )   # small subplot (1st row, 3rd column)
            axs.append(fig.add_subplot(gs[1,2]))

            idx,bbox = self.dset.get_window_index(t,x,y,dataset_key)
            # print(idx,bbox)
            
            # plot the full image
            rect = patches.Rectangle((bbox[0],bbox[2]), bbox[1]-bbox[0],bbox[3]-bbox[2],
                                    linewidth=2, edgecolor='r', facecolor='none')
            axs[0].set_title('Full Image')
            axs[0].add_patch(rect)
            imagemap(axs[0],image[t],colorbars=True);
            if scalebar_:
                # adds a scalebar to the figure
                cropped_scalebar = self.scalebar_.copy()
                cropped_scalebar["width"]*=(image[0].shape[0]/raw.shape[0])
                add_scalebar(axs[0], cropped_scalebar);

            axs[1].set_title('FFT tile')
            imagemap(axs[1],h[dataset_key][idx],colorbars=True);

            axs[2].set_title('Image tile')
            imagemap(axs[2],image[t,bbox[2]:bbox[3],bbox[0]:bbox[1]],colorbars=True);

            if self.printer is not None:
                self.printer.savefig(fig,
                                        f'{img_name[0]}_{img_name[1]}_raw', 
                                        tight_layout=False,showfig=True)

            plt.tight_layout()
            plt.show()


    def get_theta(self,rotation):
        """get the theta value from a n x 2 x 3 rotation matrix

        Args:
            rotation (array): n x 2 x 3 affine matrix

        Returns:
            array: array of size n with theta in radians
        """        
        acos = np.arccos(rotation[:,0,0])
        asin = np.arcsin(rotation[:,0,1])
        theta = asin.copy()

        # asin, acos both + means the angle is accurate

        # asin, acos both - means 3rd quadrant
        theta[ np.intersect1d(np.argwhere(acos<0), np.argwhere(asin<0)) ] *= -1 
        theta[ np.intersect1d(np.argwhere(acos<0), np.argwhere(asin<0)) ] += -np.pi

        # asin +, acos - means 2nd quadrant
        theta[ np.intersect1d(np.argwhere(acos<0), np.argwhere(asin>0)) ] *= -1
        theta[ np.intersect1d(np.argwhere(acos<0), np.argwhere(asin>0)) ] += np.pi/2
        
        return theta
    

    def real_space_affine(self,rotations,translations,scalings):
        """Calculate the rotation in degrees, absoluate value of scaling, and absolute value of translation

        Args:
            rotation (numpy array): 2 x 3 affine matrix
            translation (numpy array):  2 x 3 affine matrix
            scaling (numpy array):  2 x 3 affine matrix

        Returns:
            (tuple): xyscaling, rotations, translations
        """
        
        n = int((rotations.shape[0]/self.dset.t_len)**0.5)

        rotations_ = self.get_theta(rotations)
        xyscaling_ = np.sqrt(scalings[:,0,0]**2 +\
                             scalings[:,1,1]**2)
        translations_ = np.sqrt(translations[:,0,0]**2 +\
                             translations[:,1,1]**2)
        
        return rotations_,xyscaling_,translations_
    

    def find_nearest(self, array, value, averaging_number):
        """Finds the nearest value in an array

        This is useful when generating data from the embedding space.

        Args:
            array (array): embedding values
            value (array): current value
            averaging_number (int): how many spectra to use for averaging in the embedding space

        Returns:
            list : list of indexes to use for averaging
        """

        idx = (np.abs(array-value)).argsort()[0:averaging_number]
        return idx


    def predictor(self, values):
        """Computes the forward pass of the autoencoder

        Args:
            values (array): input values to predict

        Returns:
            array: predicted output values
        """
        with torch.no_grad():
            values = torch.from_numpy(np.atleast_2d(values))
            values = self.model(values.float())
            values = values.detach().numpy()
            return values


    def generator_images(self,
                         embedding=None,
                         folder_name='',
                         ranges=None,
                         generator_iters=200,
                         averaging_number=100,
                         graph_layout=[2, 2],
                         shape_=[256, 256, 256, 256],
                         **kwargs
                         ):
    #     """Generates images as the variables traverse the latent space

    #     Args:
    #         embedding (tensor, optional): embedding to predict with. Defaults to None.
    #         folder_name (str, optional): name of folder where images are saved. Defaults to ''.
    #         ranges (list, optional): sets the range to generate images over. Defaults to None.
    #         generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
    #         averaging_number (int, optional): number of embeddings to average. Defaults to 100.
    #         graph_layout (list, optional): layout parameters of the graph. Defaults to [2, 2].
    #         shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
    #     """

    #     # sets the kwarg values
    #     for key, value in kwargs.items():
    #         exec(f'{key} = value')

    #     # sets the channels to use in the object
    #     if "channels" in kwargs:
    #         self.channels = kwargs["channels"]

    #     # gets the embedding if a specific embedding is not provided
    #     if embedding is None:
    #         embedding = self.model.embedding

    #     # makes the folder to save the images
    #     folder = make_folder(
    #         self.printer.basepath + f"generator_images_{folder_name}/")

    #     # loops around the number of iterations to generate
    #     for i in tqdm(range(generator_iters)):

    #         # builds the figure
    #         fig, ax = layout_fig(graph_layout[0], graph_layout[1], **kwargs)
    #         ax = ax.reshape(-1)

    #         # loops around all of the embeddings
    #         for j, channel in enumerate(self.channels):

    #             if ranges is None:
    #                 ranges = np.stack((np.min(self.model.embedding, axis=0),
    #                                    np.max(self.model.embedding, axis=0)), axis=1)

    #             # linear space values for the embeddings
    #             value = np.linspace(ranges[j][0], ranges[j][1],
    #                                 generator_iters)

    #             # finds the nearest point to the value and then takes the average
    #             # average number of points based on the averaging number
    #             idx = find_nearest(
    #                 self.model.embedding[:, channel],
    #                 value[i],
    #                 averaging_number)

    #             # computes the mean of the selected index
    #             gen_value = np.mean(self.model.embedding[idx], axis=0)

    #             # specifically updates the value of the embedding to visualize based on the
    #             # linear spaced vector
    #             gen_value[channel] = value[i]

    #             # generates the loop based on the model
    #             generated = self.model.generate_spectra(gen_value).squeeze()

    #             imagemap(ax[j], generated.reshape(
    #                 shape_[0], shape_[1]), clim=[0, 6], **kwargs)
    #             ax[j].plot(3, 3, marker='o', markerfacecolor=self.cmap(
    #                 (i + 1) / generator_iters))

    #             axes_in = ax[j].inset_axes([0.55, 0.02, 0.43, 0.43])

    #             # plots the imagemap and formats
    #             imagemap(axes_in, self.model.embedding[:, channel].reshape(
    #                 shape_[2], shape_[3]), clim=ranges[j], colorbars=False)

    #         if self.printer is not None:
    #             self.printer.savefig(fig,
    #                                  f'{i:04d}_maps', tight_layout=False, basepath=folder)

    #         plt.close(fig)
        return


    def layout_embedding_affine(self, embedding, rotation, translation, scaling,
                                save_folder='.', 
                                save_figure=False,    
                                divider_=False, 
                                labelfigs_=True,
                                scalebar_=True,
                                channels = None,
                                format="%.1e",
                                **kwargs):
        """function to plot the embeddings of the data

        Args:
            embedding (numpy array): embedding created by model encoder
            rotation (numpy array): rotation created by model encoder
            translation (numpy array): translation created by model encoder
            scaling (numpy array): scaling created by model encoder
            save_folder (str, optional): where to save embedding and affines. Defaults to '.'.
            save_figure (bool, optional): unused. Defaults to False.
            divider_ (bool, optional): unused. Defaults to False.
            labelfigs_ (bool, optional): unused. Defaults to True.
            scalebar_ (bool, optional): unused. Defaults to True.
            channels (_type_, optional): _description_. Defaults to None.
        """        
        # def layout_embedding_images(embedding,rotation,translation,scaling,
        #                     t_len,a,b,
                            # combined,f,date,temps):

        xyscaling, rotations, translations = self.real_space_affine(rotation, translation, scaling)
        elim = embedding.min()*1.05,embedding.max()*0.95
        if elim[0]==elim[1]: elim = (0,1)
        folder = make_folder(f'{save_folder}')
        elim = embedding.min()*1.05,embedding.max()*0.95
        if elim[0]==elim[1]: elim = (0,1)
        folder = make_folder(f'{save_folder}')
        n = int((embedding.shape[0]/self.dset.t_len)**0.5)
        
        with h5py.File(self.dset.combined_h5_path,'a') as h:
            # make images of embeddings+original image
            for t,temp in enumerate(tqdm(self.dset.temps)):
                title = re.split('/|\.',temp)[-3]+' '+re.split('/|\.',temp)[-2]+'$^{\circ}$C'
                image = h['All_filtered'][t]
                idx = (t*n*n,(t+1)*n*n)

                plt.ioff()
                figg = plt.figure();
                gs = figg.add_gridspec(4,6);
                axsg = []
                axsg.append( figg.add_subplot(gs[:,:2]) ) # large subplot (2 rows, 2 columns)
                axsg.append( figg.add_subplot(gs[:,2:4]) ) # large subplot (2 rows, 2 columns)
                axsg.append( figg.add_subplot(gs[:,4:6]) ) # large subplot (2 rows, 2 columns)
                figg.suptitle(title,y=0.9);

                axsg[0].set_title('Full Image',fontsize='x-small');
                imagemap(axsg[0], image, **kwargs)
                if self.scalebar_ is not None:
                    # adds a scalebar to the figure
                    add_scalebar(axsg[0], self.scalebar_)

                # Embeddings
                axsg[1].set_title('Embeddings',fontsize='x-small');
                axsg[1].axis('off');
                if channels is None:
                    channels = range(embedding.shape[1])
                fig, axs = layout_fig(len(channels), mod=2, **kwargs)
                for i,c in enumerate(channels):
                    imagemap(axs[i], embedding[idx[0]:idx[1], c].reshape((n,n)).T, 
                                divider_=False,
                                colorbars=False,
                                clim = elim, 
                                format=format,
                                **kwargs)
                plot_into_graph(axsg[1],fig,clim=elim,format=format)

                # Transforms
                axsg[2].set_title('Transforms',fontsize='x-small');
                axsg[2].axis('off');
                fig, axs = layout_fig(6, mod=2);
                to_plot = [scaling[idx[0]:idx[1],0,0].T, scaling[idx[0]:idx[1],1,1].T,
                        xyscaling[idx[0]:idx[1]].T, rotations[idx[0]:idx[1]].T,
                        translation[idx[0]:idx[1],0,2].T, translation[idx[0]:idx[1],1,2].T]
                for i,data in enumerate(to_plot):
                    if i==3: 
                        imagemap(axs[i], data.reshape((n,n)).T,divider_=False, 
                                colorbars=True, cmap_='twilight',clim = (-np.pi,np.pi),
                                cbar_number_format="%1.2f")
                    else: imagemap(axs[i], data.reshape((n,n)).T,divider_=False, 
                                colorbars=True, cbar_number_format="%1.2f")
                plot_into_graph(axsg[2],fig,colorbar_=False,format=format)

                figg.tight_layout();
                # figg.savefig(f'{folder}/{t:02d}.png',facecolor='white',dpi=20); 

                if self.printer is not None:
                    make_folder(f'{self.printer.basepath}/{folder}')
                    self.printer.savefig(figg,f'{folder}/{t:02d}', 
                                         tight_layout=True,showfig=True,
                                         verbose=False)
                plt.close('all')
                plt.clf()


    def ezmask(self,flat_image,thresh,eps=2):
        """creates mask from flattened 2d image and performs binary dilation/erosion

        Args:
            image (numpy array): stack of square images flattened on 0-axis
            thresh (float): mask cutoff intensity
            eps (int, optional): radius of dilation/erosion. Defaults to 2.

        Returns:
            numpy array: binary 2d array of mask
        """        
        n = int(flat_image.shape[0]**0.5)
        mask = flat_image > thresh
        # mask = morphology.binary_closing(mask)
        mask = morphology.binary_opening(mask.reshape((n,n)),morphology.disk(eps))
        mask = morphology.binary_closing(mask.reshape((n,n)),morphology.disk(eps))
        return mask
    

    def div_except(self,embedding,t,c,emb_size=8):
        """divide channel c by all the other channels

        Args:
            embedding (numpy array): embedding of size (n*n x num_channels), where n is the dimension of reshaped image
            t (int): temperature image take at
            c (int): embedding channel to keep

        Returns:
            numpy array: cleaned channel image of shape (n,n)
        """        
        indices = list(range(emb_size))
        indices.pop(c)
        n = int((embedding.shape[0]/self.dset.t_len)**0.5)
        return embedding[t*n*n:(t+1)*n*n:,indices].sum(axis=(0))/(emb_size-1)
    

    def make_mask(self,embedding, t, c, dataset_key, d=None,
                  plot_=True, save_folder=None, err_std=0, eps=2):
        """makes figure with image, warp, histogram, and mask (if specified).

        Args:
            embedding (numpy array): 
            t (int): temperature
            c (int): 2D embedding channel to make a mask of.
            d (int, optional): divide target channel another embedding. Defaults to None.
            plot_ (bool, optional): whether to show plot. Defaults to True.
            save_folder (str, optional): where to save the image. Defaults to None.
            err_std (int, optional): number of std of threshold to determine error range for mask. Defaults to 0.

        Returns:
            numpy array: 2d binary mask of the given embedding channel
            OR
            tuple of three numpy array: three 2d binary masks of the the given embedding channel
                                    and uncertainty defined by err_std argument
                                    (mask, lower mask, upper mask)
        """
        length = 4
        with h5py.File(self.dset.combined_h5_path,'a') as h:
            logdata = h[dataset_key]
            # print(logdata.shape)
            logdata = h[dataset_key]
            # print(logdata.shape)
            n = int((logdata.shape[0]/self.dset.t_len)**0.5)
            # step2 = int((embedding.shape[0]/self.dset.t_len))
            orig_images = h['All_filtered']

            im = embedding[t*n*n:(t+1)*n*n,c]
            # print(im.shape)
            # print(im.shape)
            image=im
            if image.max==0:
                mask = image
                thresh=0
            else:
                if d!=None:
                    if d=='All': 
                        div = self.div_except(embedding,t,c) 
                    else: 
                        div = embedding[t*n*n:(t+1)*n*n, d] +\
                            embedding[int(self.dset.t_len/2)*n*n:(int(self.dset.t_len/2+1))*n*n, d]
                        # div = embedding[t,:,:,d].sum(axis=0) 
                    image = image/(div+1)
                    image = image-div
                    image[image<0] = 0
                    length = 5
                    
                if image.max()==0: # if its 0
                    mask=image
                    thresh=0
                else:
                    thresh = threshold_otsu(embedding[:,c])
                    mask = self.ezmask(image,thresh).astype(int)
                    
            if err_std>0:
                if image.max()==0: 
                    mask0,mask1 = mask,mask
                else:
                    mask0 = self.ezmask(image,max([0,thresh-im.std()*err_std])).astype(int)
                    mask1 = self.ezmask(image,thresh+im.std()*err_std).astype(int)

            if plot_==True:
                ## make figure
                fig,axes = subfigures(3,2);

                fig.set_figheight(8)
                fig.set_figwidth(8)
                temp = self.dset.temps[t].split('/')[-2]+ ' '+self.dset.temps[t].split('/')[-1].split('.')[-2]
                # fig.suptitle(f'{self.dset.combined_name} at {temp}$^\circ$C')

                axes[0].set_title(f'{self.dset.combined_name} at {temp}$^\circ$C')
                imagemap(axes[0],orig_images[t])

                axes[1].set_title(f'Embedding channel {c}')
                imagemap(axes[1],im.reshape((n,n)).T)

                axes[2].set_title(f'Histogram for Embedding')
                axes[2].hist(image.flatten(),bins=50);
                axes[2].axvline(thresh, color='k', ls='--')
                if err_std>0:
                    axes[2].axvline(thresh+im.std()*err_std,color='r',ls='--')
                    axes[2].axvline(thresh-im.std()*err_std,color='r',ls='--')
                axes[2].set_aspect('auto')

                axes[3].set_title(f'Mask')
                if err_std>0: 
                    imagemap(axes[3], (mask+mask0+mask1).T, clim=(0,3), str='%.0d')
                else: 
                    imagemap(axes[3], mask.T, clim=(0,1), str='%.0d')

                if d!=None:
                    axes[4].set_title('Cleaned')
                    imagemap(axes[4],image.T)
                    fig.delaxes(axes[5])
                else: 
                    fig.delaxes(axes[5])
                    fig.delaxes(axes[4])

                if self.printer is not None:
                    make_folder(f'{self.printer.basepath}/{save_folder}')
                    self.printer.savefig(fig,f'{save_folder}/t_{t:02d}_c_{c:02d}', 
                                         tight_layout=True,showfig=True,
                                         verbose=False)
                    
                fig.tight_layout()
                plt.show()
                plt.clf()

        if err_std>0:
            if image.max()==0: return image,image,image
            return mask,mask0,mask1
        else:
            return mask


    def graph_relative_area(self,
                            embedding,
                            dataset_key,
                            channels=range(8),
                            masked=False,
                            clean_div=None,
                            smoothing=None,
                            legends=None,
                            plot=True,
                            err_std=0,
                            save_folder='relative_areas'):
        """calculated relative areas of domains in given embeddings and creates graph

        Args:
            embedding (numpy array): embedding
            dataset_key (str): key to input dataset in h5 file
            channels (list, optional): Othewise, specify indices. Default is all 8 channels
            masked (bool, optional): whether to calculate relative areas with binary mask, 
                or with relative image intensities. Defaults to False.
            clean_div (list, optional): Channels that can be used to eliminate stray signal 
                in selected embedding channels. Defaults to None.
            smoothing (int, optional): convolution (smoothing) factor. Must be odd for odd number of temps, 
                and even for even length.. Defaults to None.
            legends (list of str, optional): Domain labels. Defaults to None.
            plot (bool, optional): whether to and save plot. Defaults to True.
            err_std (int, optional): std to find uncertainty in masks. Defaults to 0.
            save_folder (str, optional): name of folder to save it in. Defaults to 'relative_areas.

        Returns:
            dict: legends as keys and list of relative areas at each temperature
        """        
        
        rel_areas_emb = np.zeros((len(channels),self.dset.t_len))
        rel_areas_err = np.zeros((len(channels),self.dset.t_len,2))
        n = int((embedding.shape[0]/self.dset.t_len)**0.5)
        
        for t in tqdm(range(self.dset.t_len)):
            for i,c in enumerate(channels):
                im=embedding[t*n*n:(t+1)*n*n,c]
                if im.max()==0:
                    mask = np.zeros(im.shape)
                else:   
                    if clean_div!=None: 
                        if clean_div=='All': 
                            div = self.div_except(t,c)
                        else: 
                            div = embedding[t*n*n:(t+1)*n*n,clean_div[i]]
                        #divide by warp
                        im = im/(embedding[int(self.dset.t_len/2)*n*n:\
                                           int(self.dset.t_len/2+1)*n*n,2] + div+1)
                        im = im/(div+1)

                    if clean_div==None: 
                        mask = self.make_mask(embedding,t,c,dataset_key,
                                              plot_=False,err_std=err_std)
                    elif clean_div=='All': 
                        mask = self.make_mask(embedding,t,c,dataset_key,d='All',
                                              plot_=False,err_std=err_std)
                    else: 
                        mask = self.make_mask(embedding,t,c,dataset_key,d=clean_div[i],
                                              plot_=False,err_std=err_std)
                
                if masked:
                    if err_std==0: rel_areas_emb[i][t] = mask.mean()
                    else:
                        rel_areas_emb[i][t] = mask[0].mean()
                        rel_areas_err[i][t][0] = mask[1].mean()
                        rel_areas_err[i][t][1] = mask[2].mean()
                else:   
                    # rel_areas_emb[i][t] = im.mean()
                    if err_std==0: 
                        rel_areas_emb[i][t] = (im*mask).mean()
                    else:
                        rel_areas_emb[i][t] = (im*mask[0]).mean()
                        rel_areas_err[i][t][0] = (im*mask[1]).mean()
                        rel_areas_err[i][t][1] = (im*mask[2]).mean()

        # smooth down the curves
        smooth_list=[]
        smooth_err=[[],[]]
        for i,r in enumerate(rel_areas_emb):
            if smoothing!=None: 
                x=int((smoothing-1)/2)
                rel_area_smooth = np.convolve(np.pad(r,x,mode='edge'), 
                                              np.ones(smoothing)/smoothing,'valid' )
                rel_area_smooth_err0 = np.convolve(np.pad(rel_areas_err[i,:,0],x,mode='edge'), 
                                                   np.ones(smoothing)/smoothing,'valid')
                rel_area_smooth_err1 = np.convolve(np.pad(rel_areas_err[i,:,1],x,mode='edge') ,
                                                   np.ones(smoothing)/smoothing,'valid')
                # rel_area_smooth = resize(rel_area_smooth.reshape(1,-1),(1,self.t_len)).flatten()
                smooth_list.append(rel_area_smooth)
                smooth_err[0].append(rel_area_smooth_err0)
                smooth_err[1].append(rel_area_smooth_err1)
            else: 
                smooth_list.append(r)
                if err_std:
                    smooth_err[0].append(rel_areas_err[i])
                    smooth_err[1].append(rel_areas_err[i])

        if plot==True:
            temp_labels = [temp.split('/')[-1].split('.')[0] for temp in self.dset.temps]
            # fonts = {'size'   : 16}
            # xticks = {'labelsize':10,}
            # yticks = {'labelsize':10,}
            # plt.rc('font',**fonts)
            # plt.rc('xtick',**xticks)
            # plt.rc('ytick',**yticks)
            # plt.rc('axes',**{'labelsize':10,
            #                  'titlesize':'large'})

            t_half=int(self.dset.t_len/2)
            x=np.linspace(0,self.dset.t_len-1,self.dset.t_len)
            new_labels=[]
            wanted_labels = ['20','23','120',f'{temp_labels[t_half]}']
            for val in temp_labels:
                if val in wanted_labels: new_labels.append(val)
                else: new_labels.append('')
                
            plt.figure(figsize=(4,4),dpi=400)   
            plt.xticks(x, new_labels)
            plt.ylabel('Area Fraction')
            plt.xlabel('Temperature ($^\circ$C)')
            plt.text(t_half-int(t_half/4), 0.6, '+$\Delta$T')
            plt.text(t_half+int(t_half/10), 0.6, '-$\Delta$T')
            plt.suptitle(f'Relative Areas of {self.dset.combined_name}')
            
            for i,r in enumerate(smooth_list): 
                plt.plot(x,r,'-o',linewidth=3,markersize=4);
                if err_std>0: # fill error bars
                    plt.fill_between(x,smooth_err[0][i],smooth_err[1][i],
                                     alpha=0.25,label='_nolegend_')
                plt.ylim(0,1)
                
            plt.axvline(int(t_half), color='k', ls='--')
            plt.tight_layout()
            if legends!=None: 
                lgd = plt.legend(legends,frameon=1)
                frame = lgd.get_frame()
                frame.set_alpha(0.5)
                frame.set_facecolor('white')


        if save_folder!=None: 
            folder=make_folder(f'{self.printer.basepath}/{save_folder}')
            plt.savefig(f'{folder}/{self.dset.combined_name}.png',facecolor='white');
        
        plt.show();
        plt.close('all')
        # print('line2 changed')
        return dict(zip(legends,smooth_list))

# TODO:
# widget to see all temps for 1) raw 2) filtered 3) unfiltered
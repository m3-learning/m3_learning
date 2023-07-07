import matplotlib.pyplot as plt
from m3_learning.viz.layout import imagemap, add_scalebar
from m3_learning.util.file_IO import make_folder
from m3_learning.viz.layout import layout_fig
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import io
import PIL
from skimage import morphology
from skimage.filters import threshold_otsu

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
        """Initialization of the Viz class
        """

        self.printer = printer
        self.labelfigs_ = labelfigs_
        self.scalebar_ = scalebar_
        self.cmap = plt.get_cmap(color_map)
        self.channels = channels
        self.dset = dset


    def view_raw(self, img_name):
        data = self.dset.get_raw_img(*img_name)
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


    def view_window(self,img_name,x,y):
        '''
        Plot the filtered image, transform, and window
        
        Args:
            img_name (list): ['condition','temperature']
            x (int): x tile
            y (int): y tile
            name (str): name given to preprocessing h5 file, which is named '(name)_preprocessed.h5'
            condition (str): 'Ramp_up' or 'Ramp_Down'
        '''
        h = self.dset.open_combined_h5()
        image = h['All_filtered']

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

        idx,bbox = self.dset.get_window_index(t,x,y)
        # print(idx,bbox)
        
        # plot the full image
        rect = patches.Rectangle((bbox[0],bbox[2]), bbox[1]-bbox[0],bbox[3]-bbox[2],
                                 linewidth=2, edgecolor='r', facecolor='none')
        axs[0].set_title('Full Image')
        axs[0].add_patch(rect)
        imagemap(axs[0],image[t],colorbars=True)

        axs[1].set_title('FFT tile')
        imagemap(axs[1],h['windows']['windows_logdata'][idx],colorbars=True)

        axs[2].set_title('Image tile')
        imagemap(axs[2],image[t,bbox[2]:bbox[3],bbox[0]:bbox[1]],colorbars=True)

        plt.tight_layout()
        plt.show()
        plt.close('all')


    def get_theta(self,rotation):
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
            rotation (_type_): _description_
            translation (_type_): _description_
            scaling (_type_): _description_

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
       

    def plot_into_graph(self,axg,fig):
        """Given an axes and figure, it will convert the figure to an image and plot it in

        Args:
            axg (_type_): _description_
            fig (_type_): _description_
        """        
        img_buf = io.BytesIO();
        fig.savefig(img_buf,bbox_inches='tight',format='png');
        im = PIL.Image.open(img_buf);
        axg.imshow(im);
        img_buf.close()


    def layout_embedding_affine(self, embedding, rotation, translation, scaling,
                                save_folder='.', 
                                save_figure=False,    
                                divider_=False, 
                                labelfigs_=True,
                                scalebar_=True,
                                channels = None,
                                **kwargs):
        """function to plot the embeddings of the data
        """            
        # def layout_embedding_images(embedding,rotation,translation,scaling,
        #                     t_len,a,b,
                            # combined,f,date,temps):

        xyscaling, rotations, translations = self.real_space_affine(rotation, translation, scaling)
        elim = embedding.max()*0.05,embedding.min()*0.95
        folder = make_folder(f'{save_folder}/embedding_affine_maps')
        h = self.dset.open_combined_h5()
        n = int((embedding.shape[0]/self.dset.t_len)**0.5)

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
            figg.suptitle(title);

            axsg[0].set_title('Full Image');
            imagemap(axsg[0], image, **kwargs)

            # Embeddings
            axsg[1].set_title('Embeddings');
            axsg[1].axis('off');
            if channels is None:
                channels = range(embedding.shape[1])
            fig, axs = layout_fig(len(channels), mod=2, **kwargs)
            for i,c in enumerate(channels):
                imagemap(axs[i], embedding[idx[0]:idx[1], c].reshape((n,n)), 
                            divider_=False, colorbars=(i==len(channels)-1),
                            clim = elim, **kwargs)
            self.plot_into_graph(axsg[1],fig)

            # Transforms
            axsg[2].set_title('Transforms');
            axsg[2].axis('off');
            fig, axs = layout_fig(6, mod=2);
            to_plot = [scaling[idx[0]:idx[1],0,0].T, scaling[idx[0]:idx[1],1,1].T,
                       xyscaling[idx[0]:idx[1]].T, rotations[idx[0]:idx[1]].T,
                       translation[idx[0]:idx[1],0,2].T, translation[idx[0]:idx[1],1,2].T]
            for i,data in enumerate(to_plot):
                if i==3: 
                    imagemap(axs[i], data.reshape((n,n)),divider_=False, 
                             colorbars=True, cmap_='twilight')
                else: imagemap(axs[i], data.reshape((n,n)),divider_=False, 
                               colorbars=True)
            self.plot_into_graph(axsg[2],fig)

            figg.tight_layout();
            figg.savefig(f'{folder}/{t:02d}.png',facecolor='white'); 
            plt.close('all')

        h.close()

    def ezmask(self,image,thresh,eps=2):
        n = int(image.shape[0]**0.5)
        mask = image > thresh
        # mask = morphology.binary_closing(mask)
        mask = morphology.binary_opening(mask.reshape((n,n)),morphology.disk(eps))
        mask = morphology.binary_closing(mask.reshape((n,n)),morphology.disk(eps))
        return mask
    

    def div_except(self,embedding,t,c):
        indices = list(range(8))
        indices.pop(c)
        n = int((embedding.shape[0]/self.dset.t_len)**0.5)
        return embedding[t*n*n:(t+1)*n*n:,indices].sum(axis=(0))/7
    

    def make_mask(self,embedding, t, c, d=None, 
                  plot_=True, save_folder=None,err_std=0,eps=2):
        """makes figure with image, warp, histogram, and mask (if specified).

        Args:
            t (_type_): temperature
            c (_type_): 2D embedding channel to make a mask of.
            d (_type_, optional): divide target channel another embedding. Defaults to None.
            plot_ (bool, optional): whether to show plot. Defaults to True.
            save_folder (_type_, optional): where to save the image. Defaults to None.
            err_std (int, optional): number of std of threshold to determine error range for mask. Defaults to 0.

        Returns:
            _type_: _description_
        """
        length = 4
        h = self.dset.open_combined_h5()
        logdata = h['windows']['windows_logdata']
        n = int((logdata.shape[0]/self.dset.t_len)**0.5)
        # step2 = int((embedding.shape[0]/self.dset.t_len))
        orig_images = h['All_filtered']

        im = embedding[t*n*n:(t+1)*n*n,c]
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
            fig,axes = layout_fig(length, mod=2)

            fig.set_figheight(10)
            fig.set_figwidth(10)
            temp = self.dset.temps[t].split('/')[-2]+ ' '+self.dset.temps[t].split('/')[-1].split('.')[-2]
            fig.suptitle(f'{self.dset.combined_name} at {temp}$^\circ$C')

            axes[0].set_title('Original Image')
            imagemap(axes[0],orig_images[t])

            axes[1].set_title(f'Embedding channel {c}')
            imagemap(axes[1],im.T)

            axes[2].set_title(f'Histogram for Embedding')
            axes[2].hist(image.flatten(),bins=50)
            axes[2].axvline(thresh, color='k', ls='--')
            if err_std>0:
                axes[2].axvline(thresh+im.std()*err_std,color='r',ls='--')
                axes[2].axvline(thresh-im.std()*err_std,color='r',ls='--')
            axes[2].set_aspect('auto')

            axes[3].set_title(f'Mask')
            if err_std>0: 
                imagemap(axes[3], (mask+mask0+mask1).T, clim=(0,3))
            else: 
                imagemap(axes[3], mask.T, clim=(0,1))

            if d!=None:
                axes[4].set_title('Cleaned')
                imagemap(axes[4],image.T)

            if save_folder!=None:
                plt.savefig(save_folder)
        plt.show()
        plt.clf()
        h.close()

        if err_std>0:
            if image.max()==0: return image,image,image
            return mask,mask0,mask1
        else:
            return mask

# TODO:
# widget to see all temps for 1) raw 2) filtered 3) unfiltered

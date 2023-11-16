from util.file_IO import make_folder
from viz.layout import layout_fig
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from viz.layout import layout_fig, imagemap, labelfigs, find_nearest, add_scalebar
from os.path import join as pjoin
from viz.nn import embeddings as embeddings_
from viz.nn import affines as affines_
import glob
import os
import collections

class Viz:

    """Visualization class for the STEM_AE class
    """

    def __init__(self,
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

    @property
    def model(self):
        """model getter

        Returns:
            obj: neural network model
        """
        return self._model

    @model.setter
    def model(self, model):
        """Model setter

        Args:
            model (object): neural network model
        """
        self._model = model

    @property
    def channels(self):
        """channel that are visualized getter

        Returns:
            list: channels that are visualized
        """

        return self._channels

    @channels.setter
    def channels(self, channels):
        """channel that are visualized setter

        Args:
            channels (list): channels that are visualized
        """

        if channels == None:
            # if none is given, all channels are visualized
            try:
                self._channels = range(self.model.embedding.shape[1])
            except:
                self._channels = None
        else:
            self._channels = channels

    def STEM_raw_and_virtual(self, dataset,ind,
                             bright_field_=True,
                             dark_field_=True,
                             scalebar_=True,
                             using_panel=False,
                             **kwargs):
        """visualizes the raw STEM data and the virtual STEM data

        Args:
            data (np.array): raw data to visualize
            bright_field_ (list, optional): bounding box for the bright field diffraction spot. Defaults to None.
            dark_field_ (list, optional): bounding box for the dark field diffraction spot. Defaults to None.
            scalebar_ (bool, optional): determines if the scalebar is shown. Defaults to True.
            filename (string, optional): Name of the file to save. Defaults to None.
            shape_ (list, optional): shape of the original data structure. Defaults to [265, 256, 256, 256].
        """
        # ind = dataset.meta['particle_list'].where(particle)
        h=dataset.open_h5()
        data = h['processed_data'][dataset.meta['particle_inds'][ind]:
                                   dataset.meta['particle_inds'][ind+1]]
        
        # sets the number of figures based on how many plots are shown
        fig_num = 1
        if bright_field_:
            fig_num += 1
        if dark_field_:
            fig_num += 1
            
        if not using_panel: plt.ioff();
        
        # creates the figure
        fig, axs = layout_fig(fig_num, fig_num, figsize=(
                                1.5*fig_num, 1.25))
        a=0
        # plots the raw STEM data
        imagemap(axs[a], np.mean(data, axis=0), divider_=False)
        a+=1

        # plots the virtual bright field image
        if bright_field_ is not None:
            bright_field = data[:,dataset.BF_inds[ind][0],
                                  dataset.BF_inds[ind][1]]
            bright_field = bright_field.mean(axis=1)
            bright_field = bright_field.reshape(dataset.meta['shape_list'][ind][0],
                                                dataset.meta['shape_list'][ind][1])
            imagemap(axs[a], bright_field, divider_=False)
            a+=1

        # plots the virtual dark field image
        if dark_field_ is not None:
            dark_field = data
            dark_field[:,dataset.BF_inds[ind][0],
                         dataset.BF_inds[ind][1]] = 0
            
            dark_field = np.mean(dark_field,axis=(1,2))
            dark_field.reshape(dataset.meta['shape_list'][ind][0],
                               dataset.meta['shape_list'][ind][1])
            imagemap(axs[a], dark_field, divider_=False)

        # adds labels to the figure
        if self.labelfigs_:
            for j, ax in enumerate(axs):
                labelfigs(ax, j)

        if scalebar_:
            # adds a scalebar to the figure
            add_scalebar(axs[-1], self.scalebar_)
            
        if not using_panel:
            make_folder(f'{self.printer.basepath}bf_df/')
            # saves the figure
            if self.printer is not None:
                self.printer.savefig(fig, f'bf_df/{dataset.meta["particle_list"][ind]}', tight_layout=False)
        else: plt.show()
                
        plt.close('all')
        plt.clf()


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

    def embeddings(self,meta,savefolder,overwrite=False, **kwargs):
        """function to plot the embeddings of the data
        """        
        # if savefolder
        for i,p_name in enumerate(tqdm(meta['particle_list'])): # each sample
            if not overwrite: 
                existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{savefolder}*')]
                if p_name+'_embedding_maps.png' in existing:
                    print('skipping',savefolder+p_name+'_embedding_maps.png')
                    continue

            data=self.model.embedding[ meta['particle_inds'][i]:\
                                        meta['particle_inds'][i+1]]
            make_folder(self.printer.basepath+savefolder)
            # print(self.printer.basepath+savefolder)
            embeddings_(data, 
                    channels=self.channels, 
                    labelfigs_ = self.labelfigs_,
                    printer = self.printer,
                    shape_=meta['shape_list'][i],
                    name = savefolder+p_name,
                    # clim=(0,data.max()),
                    **kwargs)
            plt.clf();
            plt.close();
    
    def affines(self,meta,savefolder,overwrite=False, **kwargs):
        """function to plot the embeddings of the data
        """        
        # if savefolder
        for i,p_name in enumerate(tqdm(meta['particle_list'])): # each sample
            if overwrite: 
                existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{savefolder}*')]
                if p_name+'_affine_maps.png' in existing:
                    print('skipping',savefolder+p_name+'_embedding_maps.png')
                    continue

            affines_data=(self.model.scale_shear[ meta['particle_inds'][i]:\
                                        meta['particle_inds'][i+1]],
                          self.model.scale_shear[ meta['particle_inds'][i]:\
                                        meta['particle_inds'][i+1]],
                          self.model.scale_shear[ meta['particle_inds'][i]:\
                                        meta['particle_inds'][i+1]] )
            make_folder(self.printer.basepath+savefolder)
            affines_(affines_data, 
                    labelfigs_ = self.labelfigs_,
                    printer = self.printer,
                    shape_=meta['shape_list'][i],
                    name = savefolder+p_name,
                    # clim=(0,data.max()),
                    **kwargs)
            plt.clf();
            plt.close();
            
    def clustered_images(self, dset, cluster_list, labels,
                         save_folder = './',
                         labelfigs_ = False,
                         scalebar_ = None,
                         printer = None,
                         **kwargs):
        
        # colors = iter([plt.cm.tab20(i) for i in range(20)])
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'pink', 'gray', 'olive', 'cyan', 'navy', 'teal', 'maroon', 'silver', 'tan', 'gold', 'purple', 'moccasin', 'bisque', 'wheat', 'peachpuff', 'navajowhite', 'salmon', 'crimson', 'palevioletred', 'darksalmon', 'lightcoral', 'hotpink', 'palegoldenrod', 'plum', 'darkkhaki', 'orchid', 'thistle', 'lightgray', 'lightgreen', 'lightblue', 'lightskyblue', 'lightyellow', 'lavender', 'linen']
        n_clusters = len(np.unique(labels))
        colors = colors[:n_clusters]
        self.cmap = ListedColormap(colors,'segmented',N=n_clusters)
        
        for i,particle in enumerate(tqdm(dset.meta['particle_list'])):
            # builds the figure
            fig, axs = layout_fig(1, **kwargs)

            # loops around the channels to plot
            imagemap(axs[0], cluster_list[i],clim=(0,n_clusters-1),cmap=self.cmap,
                     cbar_number_format='%2.0e', divider_=False, **kwargs)

            # adds labels to the figure
            if labelfigs_:
                for i, ax in enumerate(axs):
                    labelfigs(ax, i)

            # adds the scalebar
            if scalebar_ is not None:
                add_scalebar(axs.flatten()[-1], scalebar_)
            
            # prints the image
            if printer is not None:
                make_folder(self.printer.basepath+save_folder)
                self.printer.savefig(fig,
                    f'./{save_folder}/{particle}', tight_layout=False)
            
            # plt.clf()
            # plt.imshow(cluster_list[i],vmin=0,vmax=19,cmap='tab20')
            # cbar = plt.colorbar()
            # cbar.set_ticks(np.arange(0,20))
            # cbar.set_ticklabels(np.arange(20))
            # plt.title(particle)
            # plt.savefig(f'./clustering_emb_affine_epoch_13/{particle}',facecolor='white',dpi=200)
            
              
    def generator_images(self,meta,generated,
                         embedding=None,
                         folder_name='',
                         ranges=None,
                         graph_layout=[2, 2],
                         shape_=[256, 256, 128, 128],
                         clim=(0.0,1.0),
                         scaler=None,
                         overwrite=False,
                         cluster_list=None,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        # sets the channels to use in the object
        if "channels" in kwargs:
            self.channels = kwargs["channels"]

        he = self.model.open_embedding_h()
        hg = self.model.open_generated_h()

        check = self.model.checkpoint.split('/')[-1][:-4]
        self.model.generated = hg[check]
        if self.model.generated.attrs['clustered']: self.cmap=plt.get_cmap('tab20')   
        else: self.cmap='viridis'   

        self.model.embedding = he[f'embedding_{check}']
        num_particles,generator_iters,emb_ch,_,_ = self.model.generated.shape

        # # gets the embedding if a specific embedding is not provided
        # if embedding is None: embedding = self.model.embedding
        make_folder(f'{self.printer.basepath}{folder_name}/')

        for p,p_name in enumerate(meta['particle_list']): # each sample
            existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{folder_name}/{p_name}/*')]
            
            print(p, p_name)
            emb = self.model.embedding[ meta['particle_inds'][p]:\
                                        meta['particle_inds'][p+1]]
            make_folder(f'{self.printer.basepath}{folder_name}/{p_name}/')
            counts = collections.Counter(cluster_list[p].flatten())
            
            # loops around the number of iterations to generate
            for i in tqdm(range(generator_iters)):
                
                if i not in counts.keys(): continue

                if not overwrite:
                    if f'{i:04d}_maps.png' in existing:
                        print('skipping',f'{folder_name}/{p_name}/{i:04d}_maps')
                        continue

                # builds the figure
                fig, ax = layout_fig(graph_layout[0]+self.model.generated.attrs['clustered'], graph_layout[1], **kwargs)
                ax = ax.reshape(-1)
                data = self.model.generated[p][i]

                try:
                    data=scaler.inverse_transform(
                        data.reshape(-1,128*128) ).reshape(-1,128,128)
                except: pass

                # loops around all of the embeddings
                for j, channel in enumerate(self.channels):

                    if ranges is None: # span this range when generating
                        ranges = np.stack((np.min(data, axis=(1,2)),
                                        np.max(data, axis=(1,2))), axis=1)
                
                    # plot the generated image
                    imagemap(ax[j], data[j], **kwargs)
                    
                    if not self.model.generated.attrs['clustered']: 
                        pt = int(shape_[-1]*0.1)
                        ax[j].plot(shape_[-1]-pt-1,pt, marker='o', markeredgewidth=0.0, markersize=3,
                                markerfacecolor=self.cmap((i+1)/generator_iters))
                        
                    axes_in = ax[j].inset_axes([0.64, 0.01, 0.35, 0.35])
                        
                    # plots the embedding and formats embedding
                    imagemap(axes_in, 
                            emb[:,channel].reshape(meta['shape_list'][p][-4], meta['shape_list'][p][-3]), 
                            colorbars=False,colormap='viridis')
                        
                    # adds labels to the figure
                    if self.labelfigs_: labelfigs(ax[j], j, size=4, text_pos="center")
       
                if self.model.generated.attrs['clustered']: 
                    # plots the clustered embedding and formats embedding
                    imagemap(ax[-1],cluster_list[p], clim=(0,19),
                            colorbars=False,colormap=self.cmap) 
                    pt = int(shape_[-3]*0.1)
                    ax[-1].plot(shape_[-3]-pt-1,pt, marker='o', markeredgewidth=0.2, markersize=4,
                            markerfacecolor=self.cmap(i))
                    
                if self.printer is not None:
                    self.printer.savefig(fig,
                                    f'{folder_name}/{p_name}/{i:04d}_maps', 
                                    tight_layout=False, 
                                    )
                plt.close(fig);
        he.close()
        hg.close()
            
  
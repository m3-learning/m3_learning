from m3_learning.util.file_IO import make_folder
from m3_learning.viz.layout import layout_fig
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ...viz.layout import layout_fig, imagemap, labelfigs, find_nearest, add_scalebar
from os.path import join as pjoin
from m3_learning.viz.nn import embeddings as embeddings_

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

    def STEM_raw_and_virtual(self, data,
                             bright_field_=None,
                             dark_field_=None,
                             scalebar_=True,
                             filename=None,
                             shape_=[256, 256, 256, 256],
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

        # sets the number of figures based on how many plots are shown
        fig_num = 1
        if bright_field_ is not None:
            fig_num += 1
        if dark_field_ is not None:
            fig_num += 1

        # creates the figure
        fig, axs = layout_fig(fig_num, fig_num, figsize=(
            1.5*fig_num, 1.25))

        # plots the raw STEM data
        imagemap(axs[0], np.mean(data.log_data.reshape(-1,
                                                       shape_[2], shape_[3]), axis=0), divider_=False)

        # plots the virtual bright field image
        if bright_field_ is not None:
            bright_field = data.data.reshape(-1, shape_[2], shape_[3])[:, bright_field_[0]:bright_field_[
                1], bright_field_[2]:bright_field_[3]]
            bright_field = np.mean(bright_field.reshape(
                shape_[0]*shape_[1], -1), axis=1).reshape(shape_[0], shape_[1])
            imagemap(axs[1], bright_field, divider_=False)

        # plots the virtual dark field image
        if dark_field_ is not None:
            dark_field = data.data.reshape(-1, shape_[2], shape_[3])[:, dark_field_[0]:dark_field_[
                1], dark_field_[2]:dark_field_[3]]
            dark_field = np.mean(dark_field.reshape(
                shape_[0]*shape_[1], -1), axis=1).reshape(shape_[0], shape_[1])
            imagemap(axs[2], dark_field, divider_=False)

        # adds labels to the figure
        if self.labelfigs_:
            for i, ax in enumerate(axs):
                labelfigs(ax, i)

        if scalebar_:
            # adds a scalebar to the figure
            add_scalebar(axs[-1], self.scalebar_)

        # saves the figure
        if self.printer is not None:
            self.printer.savefig(fig, filename, tight_layout=False)

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
        """Generates images as the variables traverse the latent space

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph. Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        # sets the channels to use in the object
        if "channels" in kwargs:
            self.channels = kwargs["channels"]

        # gets the embedding if a specific embedding is not provided
        if embedding is None:
            embedding = self.model.embedding

        # makes the folder to save the images
        folder = make_folder(
            self.printer.basepath + f"generator_images_{folder_name}/")

        # loops around the number of iterations to generate
        for i in tqdm(range(generator_iters)):

            # builds the figure
            fig, ax = layout_fig(graph_layout[0], graph_layout[1], **kwargs)
            ax = ax.reshape(-1)

            # loops around all of the embeddings
            for j, channel in enumerate(self.channels):

                if ranges is None:
                    ranges = np.stack((np.min(self.model.embedding, axis=0),
                                       np.max(self.model.embedding, axis=0)), axis=1)

                # linear space values for the embeddings
                value = np.linspace(ranges[j][0], ranges[j][1],
                                    generator_iters)

                # finds the nearest point to the value and then takes the average
                # average number of points based on the averaging number
                idx = find_nearest(
                    self.model.embedding[:, channel],
                    value[i],
                    averaging_number)

                # computes the mean of the selected index
                gen_value = np.mean(self.model.embedding[idx], axis=0)

                # specifically updates the value of the embedding to visualize based on the
                # linear spaced vector
                gen_value[channel] = value[i]

                # generates the loop based on the model
                generated = self.model.generate_spectra(gen_value).squeeze()

                imagemap(ax[j], generated.reshape(
                    shape_[0], shape_[1]), clim=[0, 6], **kwargs)
                ax[j].plot(3, 3, marker='o', markerfacecolor=self.cmap(
                    (i + 1) / generator_iters))

                axes_in = ax[j].inset_axes([0.55, 0.02, 0.43, 0.43])

                # plots the imagemap and formats
                imagemap(axes_in, self.model.embedding[:, channel].reshape(
                    shape_[2], shape_[3]), clim=ranges[j], colorbars=False)

            if self.printer is not None:
                self.printer.savefig(fig,
                                     f'{i:04d}_maps', tight_layout=False, basepath=folder)

            plt.close(fig)
            
    def embeddings(self, **kwargs):
        """function to plot the embeddings of the data
        """        
        
        embeddings_(self.model.embedding, 
                   channels=self.channels, 
                   labelfigs_ = self.labelfigs_,
                   printer = self.printer, 
                   **kwargs)
        
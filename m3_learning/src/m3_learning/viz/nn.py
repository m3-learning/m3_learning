from m3_learning.viz.layout import layout_fig, imagemap, labelfigs, add_scalebar, find_nearest
import matplotlib.pyplot as plt
import numpy as np
import torch

def embeddings(embedding, mod=4,
                   shape_=[255, 256, 256, 256], 
                   name="",
                   channels = None,
                   labelfigs_ = False,
                   scalebar_ = None,
                   printer = None,
                   **kwargs):
    """Plots the embeddings

    Args:
        embedding (_type_): _description_
        mod (int, optional): defines the number of columns in the figure. Defaults to 4.
        channels (bool, optional): specific channels to plot. Defaults to False.
        scalebar_ (dict, optional): add the scalebar. Defaults to None.
        shape_ (list, optional): shape of the initial image. Defaults to [265, 256, 256, 256].
        name (str, optional): filename. Defaults to "".
        channels (bool, optional): _description_. Defaults to False.
        labelfigs_ (bool, optional): _description_. Defaults to False.
        add_scalebar (_type_, optional): _description_. Defaults to None.
        printer (_type_, optional): _description_. Defaults to None.
    """        

    # sets the channels to use in the object
    if channels is None:
        channels = range(embedding.shape[1])


    # builds the figure
    fig, axs = layout_fig(len(channels), mod, **kwargs)

    # loops around the channels to plot
    for i in channels:
        # plots the imagemap and formats
        imagemap(axs[i], embedding[:, i].reshape(
            shape_[0], shape_[1]), divider_=False, **kwargs)

    # adds labels to the figure
    if labelfigs_:
        for i, ax in enumerate(axs):
            labelfigs(ax, i)

    # adds the scalebar
    if scalebar_ is not None:
        add_scalebar(axs.flatten()[-1], scalebar_)

    # prints the image
    if printer is not None:
        printer.savefig(fig,
            f'{name}_embedding_maps', tight_layout=False)
        
def latent_generator(
    model,
    embeddings,
    image,
    number,
    average_number,
    indx=None,
    ranges=None,
    x_values=None,
    y_scale=[-2.2, 4],
    device="cuda",
    **kwargs
):
    """Plots the generator results

    Args:
        model (PyTorch object): neural network model
        embeddings (float, array): the input embedding (or output from the encoder)
        image (array): Original image, this is used to extract the size of the embedding
        number (int): number of divisions to plot
        average_number (int): number of samples to average in the generation process
        indx (list, optional): embedding indexes to use. Defaults to None.
        ranges (float, array, optional): set the ranges for the embeddings. Defaults to None.
        x_values (array, optional): allows addition of x_values. Defaults to None.
        y_scale (list, optional): Scale of the y-axis. Defaults to [-2.2, 4].
        device (str, optional): the device where the data will be processed. Defaults to 'cuda'.
    """

    # sets the colormap
    cmap = plt.cm.viridis

    if indx is None:
        embedding_small = embeddings.squeeze()
    else:
        embedding_small = embeddings[:, indx].squeeze()

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(embedding_small.shape[1] * 2, mod=3, **kwargs)

    # plots all of the embedding maps
    for i in range(embedding_small.shape[1]):
        im = imagemap(
            ax[i], embedding_small[:, i].reshape(
                image.shape[0], image.shape[1]),
            **kwargs
        )

    # loops around the number of example loops
    for i in range(number):

        # loops around the number of embeddings from the range file
        for j in range(embedding_small.shape[1]):

            if ranges is None:
                value = np.linspace(
                    np.min(embedding_small[:, j]), np.max(
                        embedding_small[:, j]), number
                )
            else:
                # sets the linear spaced values
                value = np.linspace(0, ranges[j], number)

            idx = find_nearest(embedding_small[:, j], value[i], average_number)
            gen_value = np.mean(embeddings[idx], axis=0)
            gen_value[j] = value[i]

            # computes the generated results
            gen_value_1 = torch.from_numpy(np.atleast_2d(gen_value)).to(device)
            generated = model(gen_value_1)
            generated = generated.to("cpu")
            generated = generated.detach().numpy().squeeze()

            # plots and formats the graphs
            if x_values is None:
                ax[j + embedding_small.shape[1]].plot(
                    generated, color=cmap((i + 1) / number)
                )
            else:
                ax[j + embedding_small.shape[1]].plot(
                    x_values, generated, color=cmap((i + 1) / number)
                )

            ax[j + embedding_small.shape[1]].set_ylim(y_scale)
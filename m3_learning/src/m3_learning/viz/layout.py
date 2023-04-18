import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import torch
from matplotlib import (
    pyplot as plt,
    animation,
    colors,
    ticker,
    path,
    patches,
    patheffects,
)
import string

Path = path.Path
PathPatch = patches.PathPatch


def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds path to figure
    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    locations : numpy array
        location to position the path
    facecolor : str, optional
        facecolor of the path
    edgecolor : str, optional
        edgecolor of the path
    linestyle : str, optional
        sets the style of the line, using conventional matplotlib styles
    lineweight : float, optional
        thickness of the line
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    # extracts the vertices used to construct the path
    vertices = [
        (locations[0], locations[2]),
        (locations[1], locations[2]),
        (locations[1], locations[3]),
        (locations[0], locations[3]),
        (0, 0),
    ]
    vertices = np.array(vertices, float)
    #  makes a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor, ls=linestyle, lw=lineweight
    )
    # adds path to axes
    axes.add_patch(pathpatch)


def layout_fig(graph, mod=None, figsize=None, layout='compressed', **kwargs):
    """Utility function that helps lay out many figures

    Args:
        graph (int): number of graphs
        mod (int, optional): value that assists in determining the number of rows and columns. Defaults to None.

    Returns:
        tuple: figure and axis
    """

    # sets the kwarg values
    for key, value in kwargs.items():
        exec(f'{key} = value')

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    if mod is None:
        # Select the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7
            
    if figsize is None:
        figsize = (3 * mod, 3 * (graph // mod + (graph % mod > 0)))

    # builds the figure based on the number of graphs and a selected number of columns
    fig, axes = plt.subplots(
        graph // mod + (graph % mod > 0),
        mod,
        figsize=figsize, layout=layout
    )

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return fig, axes[:graph]


def embedding_maps(data, image, colorbar_shown=True, c_lim=None, mod=None, title=None):
    """function that generates the embedding maps

    Args:
        data (array): embedding maps to plot
        image (array): raw image used for the sizing of the image
        colorbar_shown (bool, optional): selects if colorbars are shown. Defaults to True.
        c_lim (array, optional): sets the range for the color limits. Defaults to None.
        mod (int, optional): used to change the layout (rows and columns). Defaults to None.
        title (string, optional): Adds title to the image . Defaults to None.
    """
    fig, ax = layout_fig(data.shape[1], mod)

    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels("")
            ax.set_yticklabels("")

            # adds the colorbar
            if colorbar_shown is True:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="10%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, format="%.1e")

                # Sets the scales
                if c_lim is not None:
                    im.set_clim(c_lim)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16, y=1, horizontalalignment="center")

    fig.tight_layout()


def imagemap(ax, data, colorbars=True, clim=None, divider_=True, cbar_number_format="%.1e", **kwargs):
    """pretty way to plot image maps with standard formats

    Args:
        ax (ax): axes to write to
        data (array): data to write
        colorbars (bool, optional): selects if you want to show a colorbar. Defaults to True.
        clim (array, optional): manually sets the range of the colorbars. Defaults to None.
    """

    if data.ndim == 1:
        data = data.reshape(
            np.sqrt(data.shape[0]).astype(
                int), np.sqrt(data.shape[0]).astype(int)
        )

    cmap = plt.get_cmap("viridis")

    if clim is None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data, clim=clim, cmap=cmap)

    ax.set_yticklabels("")
    ax.set_xticklabels("")
    ax.set_yticks([])
    ax.set_xticks([])

    if colorbars:
        if divider_:
            # adds the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format=cbar_number_format)
        else:
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6, width=0.05)


def find_nearest(array, value, averaging_number):
    """computes the average of some n nearest neighbors

    Args:
        array (array): input array
        value (float): value to find closest to
        averaging_number (int): number of data points to use in averaging

    Returns:
        list: list of indexes of the nearest neighbors
    """
    idx = (np.abs(array - value)).argsort()[0:averaging_number]
    return idx

def combine_lines(*args):

    lines = []
    labels = []

    for arg in args:
        # combine the two axes into a single legend
        line, label = arg.get_legend_handles_labels()
        lines += line
        labels += label

    return lines, labels


def labelfigs(
    axes, number, style="wb", loc="br", string_add="", size=8, text_pos="center"
):
    """Function that labels the figures

    Args:
        axes (axes object): axes to label
        number (int): number value of the label in letters
        style (str, optional): sets the style of the label. Defaults to 'wb'.
        loc (str, optional): sets the location of the label. Defaults to 'br'.
        string_add (str, optional): Adds a prefix string. Defaults to ''.
        size (int, optional): sets the font size. Defaults to 14.
        text_pos (str, optional): sets the position of the text. Defaults to 'center'.

    Raises:
        ValueError: The string provided for the style is not valid
    """

    # Sets up various color options
    formatting_key = {
        "wb": dict(color="w", linewidth=1.5),
        "b": dict(color="k", linewidth=0),
        "w": dict(color="w", linewidth=0),
    }

    # Stores the selected option
    formatting = formatting_key[style]

    # finds the position for the label
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_value = 0.08 * (x_max - x_min) + x_min

    # Sets the location of the label on the figure
    if loc == "br":
        y_value = y_max - 0.15 * (y_max - y_min)
        x_value = 0.15 * (x_max - x_min) + x_min
    elif loc == "tr":
        y_value = y_max - 0.9 * (y_max - y_min)
        x_value = 0.08 * (x_max - x_min) + x_min
    elif loc == "bl":
        y_value = y_max - 0.1 * (y_max - y_min)
        x_value = x_max - 0.08 * (x_max - x_min)
    elif loc == "tl":
        y_value = y_max - 0.9 * (y_max - y_min)
        x_value = x_max - 0.08 * (x_max - x_min)
    elif loc == "tm":
        y_value = y_max - 0.9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    elif loc == "bm":
        y_value = y_max - 0.1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    else:
        raise ValueError(
            "Unknown string format imported please look at code for acceptable positions"
        )

    # adds a custom string
    if string_add == "":

        # Turns to image number into a label
        if number < 26:
            axes.text(
                x_value,
                y_value,
                string.ascii_lowercase[number],
                size=size,
                weight="bold",
                ha=text_pos,
                va="center",
                color=formatting["color"],
                path_effects=[
                    patheffects.withStroke(
                        linewidth=formatting["linewidth"], foreground="k"
                    )
                ],
            )

        # allows for double letter index
        else:
            axes.text(
                x_value,
                y_value,
                string.ascii_lowercase[0] +
                string.ascii_lowercase[number - 26],
                size=size,
                weight="bold",
                ha=text_pos,
                va="center",
                color=formatting["color"],
                path_effects=[
                    patheffects.withStroke(
                        linewidth=formatting["linewidth"], foreground="k"
                    )
                ],
            )
    else:
        # writes the text to the figure
        axes.text(
            x_value,
            y_value,
            string_add,
            size=14,
            weight="bold",
            ha=text_pos,
            va="center",
            color=formatting["color"],
            path_effects=[
                patheffects.withStroke(
                    linewidth=formatting["linewidth"], foreground="k"
                )
            ],
        )


def scalebar(axes, image_size, scale_size, units="nm", loc="br"):
    """
    Adds scalebar to figures
    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    image_size : int
        size of the image in nm
    scale_size : str, optional
        size of the scalebar in units of nm
    units : str, optional
        sets the units for the label
    loc : str, optional
        sets the location of the label
    """

    # gets the size of the image
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(np.int(np.floor(x_lim[1] - x_lim[0]))), np.abs(
        np.int(np.floor(y_lim[1] - y_lim[0]))
    )
    # computes the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.int(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1], np.int(np.floor(image_size)))

    # sets the location of the scalebar
    if loc == "br":
        x_start = x_point[np.int(0.9 * image_size // 1)]
        x_end = x_point[np.int((0.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(0.1 * image_size // 1)]
        y_end = y_point[np.int((0.1 + 0.025) * image_size // 1)]
        y_label_height = y_point[np.int((0.1 + 0.075) * image_size // 1)]
    elif loc == "tr":
        x_start = x_point[np.int(0.9 * image_size // 1)]
        x_end = x_point[np.int((0.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(0.9 * image_size // 1)]
        y_end = y_point[np.int((0.9 - 0.025) * image_size // 1)]
        y_label_height = y_point[np.int((0.9 - 0.075) * image_size // 1)]

    # makes the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], "w", "k", "-", .25)

    # adds the text label for the scalebar
    axes.text(
        (x_start + x_end) / 2,
        y_label_height,
        "{0} {1}".format(scale_size, units),
        size=6,
        weight="bold",
        ha="center",
        va="center",
        color="w",
        path_effects=[patheffects.withStroke(linewidth=.5, foreground="k")],
    )


def Axis_Ratio(axes, ratio=1):
    # Set aspect ratio to be proportional to the ratio of data ranges
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    xrange = xmax - xmin
    yrange = ymax - ymin

    axes.set_aspect(ratio * (xrange / yrange))


def get_axis_range(axs):

    def get_axis_range_(ax):
        """
        Return the minimum and maximum values of a Matplotlib axis.

        Parameters:
            ax (matplotlib.axis.Axis): The Matplotlib axis object to get the range of.

        Returns:
            tuple: A tuple of the form (xmin, xmax, ymin, ymax), where xmin and xmax are the minimum and maximum values of the x axis, and ymin and ymax are the minimum and maximum values of the y axis.
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        return xmin, xmax, ymin, ymax

    for ax in axs:
        ax_xmin, ax_xmax, ax_ymin, ax_ymax = get_axis_range_(ax)
        try:
            xmin = min(xmin, ax_xmin)
            xmax = max(xmax, ax_xmax)
            ymin = min(ymin, ax_ymin)
            ymax = max(ymax, ax_ymax)
        except:
            xmin = ax_xmin
            xmax = ax_xmax
            ymin = ax_ymin
            ymax = ax_ymax

    return [xmin, xmax, ymin, ymax]


def set_axis(axs, range):
    for ax in axs:
        ax.set_xlim(range[0], range[1])
        ax.set_ylim(range[2], range[3])
        
def add_scalebar(ax, scalebar_):
    """Adds a scalebar to the figure

    Args:
        ax (axes): axes to add the scalebar to
        scalebar_ (dict): dictionary containing the scalebar information
    """

    if scalebar_ is not None:
        scalebar(ax, scalebar_['width'], scalebar_[
            'scale length'], units=scalebar_['units'])

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as transforms
from itertools import product
from matplotlib.text import Annotation
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
import PIL
import io

Path = path.Path
PathPatch = patches.PathPatch

def plot_into_graph(axg,fig,colorbar_=True,clim=None,**kwargs):
    """Given an axes and figure, it will convert the figure to an image and plot it in

    Args:
        axg (matplotlib.axes.Axes): where you want to plot the figure
        fig (matplotlib.pyplot.figure()): figure you want to put into axes
    """        
    img_buf = io.BytesIO();
    fig.savefig(img_buf,bbox_inches='tight',format='png');
    im = PIL.Image.open(img_buf);
    
    if clim!=None: ax_im = axg.imshow(im,clim=clim);
    else: ax_im = axg.imshow(im);
    
    if colorbar_:
        divider = make_axes_locatable(axg)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax_im, cax=cax,**kwargs)
    
    img_buf.close()
    
def subfigures(nrows, ncols, size=(1.25, 1.25), gaps=(.8, .33), figsize=None, **kwargs):

    if figsize is None:
        figsize = (size[0]*ncols + gaps[0]*ncols, size[1]*nrows+gaps[1]*nrows)

    # create a new figure with a size of 6x6 inches
    fig = plt.figure(figsize=figsize)

    ax = []

    for i, j in product(range(nrows), range(ncols)):
        rvalue = (nrows-1) - j
        # create the first axis with absolute position (1 inch, 1 inch) and size (2 inches, 2 inches)
        pos1 = [(size[0]*rvalue + gaps[0]*rvalue)/figsize[0], (size[1]*i + gaps[1]*i)/figsize[1],
                size[0]/figsize[0], size[1]/figsize[1]]  # transforms.Bbox.from_bounds()
        ax.append(fig.add_axes(pos1))

    ax.reverse()

    return fig, ax


def add_text_to_figure(fig, text, text_position_in_inches, **kwargs):

    # Get the figure size in inches and dpi
    fig_size_inches = fig.get_size_inches()
    fig_dpi = fig.get_dpi()

    # Convert the desired text position in inches to a relative position (0 to 1)
    text_position_relative = (
        text_position_in_inches[0] / fig_size_inches[0], text_position_in_inches[1] / fig_size_inches[1])

    # Add the text to the figure with the calculated relative position
    fig.text(text_position_relative[0],
             text_position_relative[1], text, **kwargs)


def add_box(axs, pos, **kwargs):

    xmin, ymin, xmax, ymax = pos
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
    axs.add_patch(rect)


def inset_connector(fig, ax1, ax2, coord1=None, coord2=None, **kwargs):
    if coord1 is None:
        coord1_xlim = ax1.get_xlim()
        coord1_ylim = ax1.get_ylim()

        coord1_l1 = (coord1_xlim[0], coord1_ylim[0])
        coord1_l2 = (coord1_xlim[0], coord1_ylim[1])
        coord1 = [coord1_l1, coord1_l2]

    if coord2 is None:
        coord2_xlim = ax2.get_xlim()
        coord2_ylim = ax2.get_ylim()

        coord2_l1 = (coord2_xlim[0], coord2_ylim[0])
        coord2_l2 = (coord2_xlim[0], coord2_ylim[1])
        coord2 = [coord2_l1, coord2_l2]

    for p1, p2 in zip(coord1, coord2):

        # Create a connection between the two points
        con = ConnectionPatch(xyA=p1, xyB=p2,
                              coordsA=ax1.transData, coordsB=ax2.transData, **kwargs)

        # Add the connection to the plot
        fig.add_artist(con)


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


def imagemap(ax, data, colorbars=True, clim=None, divider_=True, 
             cbar_number_format="%.1e", cmap_ = 'viridis', **kwargs):
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

    cmap = plt.get_cmap(cmap_)

    if clim is None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data, vmin=clim[0], vmax=clim[1], clim=clim, cmap=cmap)

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
            cb = plt.colorbar(im, fraction=0.046, pad=0.04,format=cbar_number_format)
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


def labelfigs(axes, number=None, style="wb",
              loc="tl", string_add="", size=8,
              text_pos="center", inset_fraction=(0.15, 0.15), **kwargs):

    # initializes an empty string
    text = ""

    # Sets up various color options
    formatting_key = {
        "wb": dict(color="w", linewidth=.75),
        "b": dict(color="k", linewidth=0),
        "w": dict(color="w", linewidth=0),
    }

    # Stores the selected option
    formatting = formatting_key[style]

    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]

    if loc == 'tl':
        x, y = xlim[0] + x_inset, ylim[1] - y_inset
    elif loc == 'tr':
        x, y = xlim[1] - x_inset, ylim[1] - y_inset
    elif loc == 'bl':
        x, y = xlim[0] + x_inset, ylim[0] + y_inset
    elif loc == 'br':
        x, y = xlim[1] - x_inset, ylim[0] + y_inset
    elif loc == 'ct':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[1] - y_inset
    elif loc == 'cb':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[0] + y_inset
    else:
        raise ValueError(
            "Invalid position. Choose from 'tl', 'tr', 'bl', 'br', 'ct', or 'cb'.")

    text += string_add

    if number is not None:
        text += number_to_letters(number)

    text_ = axes.text(x, y, text, va='center', ha='center',
                      path_effects=[patheffects.withStroke(
                      linewidth=formatting["linewidth"], foreground="k")],
                      color=formatting["color"], size=size, **kwargs
                      )

    text_.set_zorder(np.inf)


def number_to_letters(num):
    letters = ''
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1  # decrease num by 1 because we have processed the current digit
    return letters


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
    x_size, y_size = np.abs(int(np.floor(x_lim[1] - x_lim[0]))), np.abs(
        int(np.floor(y_lim[1] - y_lim[0]))
    )
    # computes the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], int(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1], int(np.floor(image_size)))

    # sets the location of the scalebar
    if loc == "br":
        x_start = x_point[int(0.9 * image_size // 1)]
        x_end = x_point[int((0.9 - fract) * image_size // 1)]
        y_start = y_point[int(0.1 * image_size // 1)]
        y_end = y_point[int((0.1 + 0.025) * image_size // 1)]
        y_label_height = y_point[int((0.1 + 0.075) * image_size // 1)]
    elif loc == "tr":
        x_start = x_point[int(0.9 * image_size // 1)]
        x_end = x_point[int((0.9 - fract) * image_size // 1)]
        y_start = y_point[int(0.9 * image_size // 1)]
        y_end = y_point[int((0.9 - 0.025) * image_size // 1)]
        y_label_height = y_point[int((0.9 - 0.075) * image_size // 1)]

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


def get_axis_pos_inches(fig, ax):
    """gets the position of the axis in inches

    Args:
        fig (matplotlib.Figure): figure where the plot is located
        ax (maplotlib.axes): axes on the plot

    Returns:
        array: the position of the center bottom of the axis in inches
    """

    # Get the bounding box of the axis in normalized coordinates (relative to the figure)
    axis_bbox = ax.get_position()

    # Calculate the center bottom point of the axis in normalized coordinates
    center_bottom_x = axis_bbox.x0 + axis_bbox.width / 2
    center_bottom_y = axis_bbox.y0

    # Convert the center bottom point from normalized coordinates to display units
    center_bottom_display = fig.transFigure.transform(
        (center_bottom_x, center_bottom_y))

    return center_bottom_display/fig.dpi


class FigDimConverter:
    """class to convert between relative and inches dimensions of a figure
    """

    def __init__(self, figsize):
        """initializes the class

        Args:
            figsize (tuple): figure size in inches
        """

        self.fig_width = figsize[0]
        self.fig_height = figsize[1]

    def to_inches(self, x):
        """Converts position from relative to inches

        Args:
            x (tuple): position in relative coordinates (left, bottom, width, height)

        Returns:
            tuple: position in inches (left, bottom, width, height)
        """

        return (x[0] * self.fig_width, x[1] * self.fig_height, x[2] * self.fig_width, x[3] * self.fig_height)

    def to_relative(self, x):
        """Converts position from inches to relative

        Args:
            x (tuple): position in inches (left, bottom, width, height)

        Returns:
            tuple: position in relative coordinates (left, bottom, width, height)
        """

        return (x[0] / self.fig_width, x[1] / self.fig_height, x[2] / self.fig_width, x[3] / self.fig_height)

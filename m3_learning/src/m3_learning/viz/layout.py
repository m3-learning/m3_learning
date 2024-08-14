from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product
import numpy as np
from matplotlib import (
    pyplot as plt,
    path,
    patches,
    patheffects,
)

Path = path.Path
PathPatch = patches.PathPatch


def subfigures(nrows, ncols, size=(1.25, 1.25), gaps=(.8, .33), figsize=None, **kwargs):
    """
    Create subfigures with specified number of rows and columns.

    Parameters:
    nrows (int): Number of rows.
    ncols (int): Number of columns.
    size (tuple, optional): Size of each subfigure. Defaults to (1.25, 1.25).
    gaps (tuple, optional): Gaps between subfigures. Defaults to (.8, .33).
    figsize (tuple, optional): Size of the figure. Defaults to None.
    **kwargs: Additional keyword arguments.

    Returns:
    fig (Figure): The created figure.
    ax (list): List of axes objects.

    """
    if figsize is None:
        figsize = (size[0]*ncols + gaps[0]*ncols, size[1]*nrows+gaps[1]*nrows)

    # create a new figure with the specified size
    fig = plt.figure(figsize=figsize)

    ax = []

    for i, j in product(range(nrows), range(ncols)):
        rvalue = (nrows-1) - j
        # calculate the position and size of each subfigure
        pos1 = [(size[0]*rvalue + gaps[0]*rvalue)/figsize[0], (size[1]*i + gaps[1]*i)/figsize[1],
                size[0]/figsize[0], size[1]/figsize[1]]
        ax.append(fig.add_axes(pos1))

    ax.reverse()

    return fig, ax


def add_text_to_figure(fig, text, text_position_in_inches, **kwargs):
    """
    Add text to a figure at a specified position.

    Parameters:
    fig (Figure): The figure to add the text to.
    text (str): The text to be added.
    text_position_in_inches (tuple): The position of the text in inches.
    **kwargs: Additional keyword arguments.

    """
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
    """
    Add a box to the axes.

    Parameters:
    axs (Axes): The axes to add the box to.
    pos (tuple): The position of the box in the form (xmin, ymin, xmax, ymax).
    **kwargs: Additional keyword arguments.

    """
    xmin, ymin, xmax, ymax = pos
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
    axs.add_patch(rect)


def inset_connector(fig, ax1, ax2, coord1=None, coord2=None, **kwargs):
    """
    Create a connection between two axes in a figure.

    Parameters:
    fig (Figure): The figure to add the connection to.
    ax1 (Axes): The first axes object.
    ax2 (Axes): The second axes object.
    coord1 (list, optional): The coordinates of the first connection point. Defaults to None.
    coord2 (list, optional): The coordinates of the second connection point. Defaults to None.
    **kwargs: Additional keyword arguments.

    """
    if coord1 is None:
        # Get the x and y limits of ax1
        coord1_xlim = ax1.get_xlim()
        coord1_ylim = ax1.get_ylim()

        # Calculate the coordinates of the first connection point
        coord1_l1 = (coord1_xlim[0], coord1_ylim[0])
        coord1_l2 = (coord1_xlim[0], coord1_ylim[1])
        coord1 = [coord1_l1, coord1_l2]

    if coord2 is None:
        # Get the x and y limits of ax2
        coord2_xlim = ax2.get_xlim()
        coord2_ylim = ax2.get_ylim()

        # Calculate the coordinates of the second connection point
        coord2_l1 = (coord2_xlim[0], coord2_ylim[0])
        coord2_l2 = (coord2_xlim[0], coord2_ylim[1])
        coord2 = [coord2_l1, coord2_l2]

    for p1, p2 in zip(coord1, coord2):
        # Create a connection between the two points
        con = ConnectionPatch(xyA=p1, xyB=p2,
                              coordsA=ax1.transData, coordsB=ax2.transData, **kwargs)

        # Add the connection to the plot
        fig.add_artist(con)


def subfigures(nrows, ncols, size=(1.25, 1.25), gaps=(.8, .33), figsize=None, **kwargs):
    """
    Create subfigures with specified number of rows and columns.

    Parameters:
    nrows (int): Number of rows.
    ncols (int): Number of columns.
    size (tuple, optional): Size of each subfigure. Defaults to (1.25, 1.25).
    gaps (tuple, optional): Gaps between subfigures. Defaults to (.8, .33).
    figsize (tuple, optional): Size of the figure. Defaults to None.
    **kwargs: Additional keyword arguments.

    Returns:
    fig (Figure): The created figure.
    ax (list): List of axes objects.

    """
    if figsize is None:
        figsize = (size[0]*ncols + gaps[0]*ncols, size[1]*nrows+gaps[1]*nrows)

    # create a new figure with the specified size
    fig = plt.figure(figsize=figsize)

    ax = []

    for i, j in product(range(nrows), range(ncols)):
        rvalue = (nrows-1) - j
        # calculate the position and size of each subfigure
        pos1 = [(size[0]*rvalue + gaps[0]*rvalue)/figsize[0], (size[1]*i + gaps[1]*i)/figsize[1],
                size[0]/figsize[0], size[1]/figsize[1]]
        ax.append(fig.add_axes(pos1))

    ax.reverse()

    return fig, ax


def add_text_to_figure(fig, text, text_position_in_inches, **kwargs):
    """
    Add text to a figure at a specified position.

    Parameters:
    fig (Figure): The figure to add the text to.
    text (str): The text to be added.
    text_position_in_inches (tuple): The position of the text in inches.
    **kwargs: Additional keyword arguments.

    """
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
    """
    Add a box to the axes.

    Parameters:
    axs (Axes): The axes to add the box to.
    pos (tuple): The position of the box in the form (xmin, ymin, xmax, ymax).
    **kwargs: Additional keyword arguments.

    """
    xmin, ymin, xmax, ymax = pos
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
    axs.add_patch(rect)


def inset_connector(fig, ax1, ax2, coord1=None, coord2=None, **kwargs):
    """
    Create a connection between two axes in a figure.

    Parameters:
    fig (Figure): The figure to add the connection to.
    ax1 (Axes): The first axes object.
    ax2 (Axes): The second axes object.
    coord1 (list, optional): The coordinates of the first connection point. Defaults to None.
    coord2 (list, optional): The coordinates of the second connection point. Defaults to None.
    **kwargs: Additional keyword arguments.

    """
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
    Create a path patch and add it to the axes.

    Parameters:
    axes (Axes): The axes to add the path patch to.
    locations (tuple): The locations of the path in the form (x1, x2, y1, y2).
    facecolor (str): The face color of the path patch.
    edgecolor (str): The edge color of the path patch.
    linestyle (str): The line style of the path patch.
    lineweight (float): The line weight of the path patch.

    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    # Extract the vertices used to construct the path
    vertices = [
        (locations[0], locations[2]),
        (locations[1], locations[2]),
        (locations[1], locations[3]),
        (locations[0], locations[3]),
        (0, 0),
    ]
    vertices = np.array(vertices, float)
    # Make a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor, ls=linestyle, lw=lineweight
    )
    # Add the path to the axes
    axes.add_patch(pathpatch)


def layout_fig(graph, mod=None, figsize=None, layout='compressed', **kwargs):
    """
    Utility function that helps lay out many figures.

    Parameters:
    graph (int): Number of graphs.
    mod (int, optional): Value that assists in determining the number of rows and columns. Defaults to None.
    figsize (tuple, optional): Size of the figure. Defaults to None.
    layout (str, optional): Layout style of the subplots. Defaults to 'compressed'.
    **kwargs: Additional keyword arguments.

    Returns:
    tuple: Figure and axes.

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
    """
    Combine the lines and labels from multiple plots into a single legend.

    Args:
        *args: Variable number of arguments representing the plots.

    Returns:
        A tuple containing the combined lines and labels.

    Example:
        lines, labels = combine_lines(plot1, plot2, plot3)
    """



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
    """
    Add labels to figures.

    Parameters:
    axes (Axes): The axes to add the labels to.
    number (int, optional): The number to be added as a label. Defaults to None.
    style (str, optional): The style of the label. Defaults to "wb".
    loc (str, optional): The location of the label. Defaults to "tl".
    string_add (str, optional): Additional string to be added to the label. Defaults to "".
    size (int, optional): The font size of the label. Defaults to 8.
    text_pos (str, optional): The position of the label text. Defaults to "center".
    inset_fraction (tuple, optional): The fraction of the axes to inset the label. Defaults to (0.15, 0.15).
    **kwargs: Additional keyword arguments.

    Returns:
    Text: The created text object.

    Raises:
    ValueError: If an invalid position is provided.

    """

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

    text_ = axes.text(x, y, text, va=text_pos, ha='center',
                      path_effects=[patheffects.withStroke(
                      linewidth=formatting["linewidth"], foreground="k")],
                      color=formatting["color"], size=size, **kwargs
                      )

    text_.set_zorder(np.inf)


def number_to_letters(num):
    """
    Convert a number to a string representation using letters.

    Parameters:
    num (int): The number to convert.

    Returns:
    str: The string representation of the number.

    """
    letters = ''
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1  # decrease num by 1 because we have processed the current digit
    return letters


def scalebar(axes, image_size, scale_size, units="nm", loc="br"):
    """
    Adds a scalebar to figures.

    Parameters:
    axes (matplotlib.axes.Axes): The axes to add the scalebar to.
    image_size (int): The size of the image in nm.
    scale_size (str): The size of the scalebar in units of nm.
    units (str, optional): The units for the label. Defaults to "nm".
    loc (str, optional): The location of the label. Defaults to "br".
    """

    # Get the size of the image
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(np.int64(np.floor(x_lim[1] - x_lim[0]))), np.abs(
        np.int64(np.floor(y_lim[1] - y_lim[0]))
    )
    # Compute the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.int64(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1], np.int64(np.floor(image_size)))

    # Set the location of the scalebar
    if loc == "br":
        x_start = x_point[np.int64(0.9 * image_size // 1)]
        x_end = x_point[np.int64((0.9 - fract) * image_size // 1)]
        y_start = y_point[np.int64(0.1 * image_size // 1)]
        y_end = y_point[np.int64((0.1 + 0.025) * image_size // 1)]
        y_label_height = y_point[np.int64((0.1 + 0.075) * image_size // 1)]
    elif loc == "tr":
        x_start = x_point[np.int64(0.9 * image_size // 1)]
        x_end = x_point[np.int64((0.9 - fract) * image_size // 1)]
        y_start = y_point[np.int64(0.9 * image_size // 1)]
        y_end = y_point[np.int64((0.9 - 0.025) * image_size // 1)]
        y_label_height = y_point[np.int64((0.9 - 0.075) * image_size // 1)]

    # Make the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], "w", "k", "-", .25)

    # Add the text label for the scalebar
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
    """
    Set the aspect ratio of the axes to be proportional to the ratio of data ranges.

    Parameters:
    axes (matplotlib.axes.Axes): The axes object to set the aspect ratio for.
    ratio (float, optional): The desired aspect ratio. Defaults to 1.

    Returns:
    None
    """
    # Set aspect ratio to be proportional to the ratio of data ranges
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    xrange = xmax - xmin
    yrange = ymax - ymin

    axes.set_aspect(ratio * (xrange / yrange))


def get_axis_range(axs):
    """
    Return the minimum and maximum values of a Matplotlib axis.

    Parameters:
        axs (list): A list of Matplotlib axis objects.

    Returns:
        list: A list of the form [xmin, xmax, ymin, ymax], where xmin and xmax are the minimum and maximum values of the x axis, and ymin and ymax are the minimum and maximum values of the y axis.
    """
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

    xmin = None
    xmax = None
    ymin = None
    ymax = None

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
    """
    Set the x and y axis limits for each axis in the given list.

    Parameters:
    axs (list): A list of matplotlib axes objects.
    range (list): A list containing the x and y axis limits in the format [xmin, xmax, ymin, ymax].

    Returns:
    None
    """
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

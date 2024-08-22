import numpy as np
import imutils
from matplotlib import (pyplot as plt, animation, colors, ticker, path, patches, patheffects)
import plotly.graph_objects as go
import pylab as pl
import scipy
from scipy import special
from scipy import signal
from scipy.signal import savgol_filter
from scipy import special
from matplotlib import animation, colors, ticker, path, patches, patheffects

class afm_substrate():
    """
    This class is designed to facilitate the analysis of an atomic force microscopy (AFM) substrate image. 
    The class includes methods for image rotation, coordinate transformation, peak detection, and step parameter calculation.
    """ 
    def __init__(self, img, pixels, size):
        """
        Initializes the AFM substrate object.

        Args:
            img (numpy.ndarray): The image to be analyzed.
            pixels (int): The number of pixels in the image.
            size (float): The size of the image in meters.
        """
        self.img = img
        self.pixels = pixels
        self.size = size
    
    def rotate_image(self, angle, colorbar_range=None, demo=True):
        """
        Rotates the image by the given angle.

        Args:
            angle (float): The angle to rotate the image in degrees.
            colorbar_range (tuple, optional): The range for the colorbar. Defaults to None.
            demo (bool, optional): If True, displays the rotated image. Defaults to True.

        Returns:
            tuple: The rotated image and its new size.
        """
        rad = np.radians(angle)
        scale = 1/(np.abs(np.sin(rad)) + np.abs(np.cos(rad)))
        size_rot = self.size * scale

        img_rot = imutils.rotate(self.img, angle=angle, scale=scale)
        h, w = img_rot.shape[:2]

        if demo:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(img_rot)
            plt.plot([0, w], [h//4, h//4], color='w')
            plt.plot([0, w], [h//2, h//2], color='w')
            plt.plot([0, w], [h*3//4, h*3//4], color='w')
            if colorbar_range:
                im.set_clim(colorbar_range) 
            plt.colorbar()
            plt.show()
        return img_rot, size_rot


    def rotate_xz(self, x, z, xz_angle):
        """
        Rotates the xz plane by the specified angle.

        Args:
            x (numpy.ndarray): The x coordinates of the image.
            z (numpy.ndarray): The z coordinates of the image.
            xz_angle (float): The angle to rotate the xz plane in degrees.

        Returns:
            tuple: Rotated x and z coordinates.
        """
        theta = np.radians(xz_angle)
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        return x_rot, z_rot

    def show_peaks(self, x, z, peaks=None, valleys=None):
        """
        Displays peaks and valleys on the plot of x and z data.

        Args:
            x (numpy.ndarray): The x-axis data.
            z (numpy.ndarray): The z-axis data (height).
            peaks (numpy.ndarray, optional): The indices of the peaks. Defaults to None.
            valleys (numpy.ndarray, optional): The indices of the valleys. Defaults to None.

        Returns:
            None
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=z, mode='lines+markers', name='Original Plot'))
        if isinstance(peaks, np.ndarray):
            marker=dict(size=8, color='red', symbol='cross')
            fig.add_trace(go.Scatter(x=x[peaks], y=z[peaks], mode='markers', marker=marker, name='Detected Peaks'))
        if isinstance(valleys, np.ndarray):
            marker=dict(size=8, color='black', symbol='cross')
            fig.add_trace(go.Scatter(x=x[valleys], y=z[valleys], mode='markers', marker=marker, name='Detected valleys'))
        fig.show()

    def slice_rotate(self, img_rot, size, j, prominence, width, xz_angle=0, demo=False):
        """
        Slices and rotates the image at a specified column and angle.

        Args:
            img_rot (numpy.ndarray): The rotated image.
            size (float): The size of the image in meters.
            j (int): The column to slice.
            prominence (float): The prominence of peaks to detect.
            width (float): The width of peaks to detect.
            xz_angle (float, optional): The angle between the x and z axes in degrees. Defaults to 0.
            demo (bool, optional): If True, displays the slice with peaks and valleys. Defaults to False.

        Returns:
            tuple: x, z coordinates and indices of peaks and valleys.
        """
        i = np.linspace(0, self.pixels-1, self.pixels)
        x = i / self.pixels * size
        z = img_rot[np.argwhere(img_rot[:, j]!=0).flatten(), j]
        x = x[np.argwhere(img_rot[:, j]!=0).flatten()]
        peak_indices, _ = signal.find_peaks(z, prominence=prominence, width=width)
        valley_indices, _ = signal.find_peaks(-z, prominence=prominence, width=width)

        if xz_angle != 0:
            x_min, x_max, z_min, z_max = np.min(x), np.max(x), np.min(z), np.max(z)
            x_norm = (x - x_min) / (x_max - x_min)
            z_norm = (z - z_min) / (z_max - z_min)

            peak_indices, _ = signal.find_peaks(z_norm, prominence=prominence, width=width)
            valley_indices, _ = signal.find_peaks(-z_norm, prominence=prominence, width=width)

            # Rotate the xz plane to level the step
            x_norm_rot, z_norm_rot = self.rotate_xz(x_norm, z_norm, xz_angle)
            x, z = x_norm_rot * (x_max - x_min) + x_min, z_norm_rot * (z_max - z_min) + z_min
        
        if demo:
            self.show_peaks(x, z, peak_indices, valley_indices)
        return x, z, peak_indices, valley_indices


    def calculate_simple(self, x, z, peak_indices, fixed_height=None, demo=False):
        """
        Calculates the height, width, and miscut of the steps in a straightforward way.

        Args:
            x (numpy.ndarray): The x-axis data.
            z (numpy.ndarray): The z-axis data (height).
            peak_indices (numpy.ndarray): The indices of the peaks.
            fixed_height (float, optional): The fixed height of the steps. Defaults to None.
            demo (bool, optional): If True, prints the calculated step properties. Defaults to False.

        Returns:
            tuple: step_heights, step_widths, miscut angles.
        """
        step_widths = np.diff(x[peak_indices])
        if fixed_height:
            step_heights = np.full(len(step_widths), fixed_height)
        else:
            step_heights = z[peak_indices[1:]] - z[peak_indices[:-1]]
        miscut = np.degrees(np.arctan(step_heights/step_widths))
        
        if demo:
            for i in range(len(step_heights)):
                print(f"Step {i+1}: Height = {step_heights[i]:.2e}, Width = {step_widths[i]:.2e}, Miscut = {miscut[i]:.3f}°")
            print('Results:')
            print(f"  Average step height = {np.mean(step_heights):.2e}, Standard deviation = {np.std(step_heights):.2e}")
            print(f"  Average step width = {np.mean(step_widths):.2e}, Standard deviation = {np.std(step_widths):.2e}")
            print(f"  Average miscut = {np.mean(miscut):.3f}°, Standard deviation = {np.std(miscut):.3f}°")
        return step_heights, step_widths, miscut

    def calculate_fit(self, x, z, peak_indices, valley_indices, fixed_height, demo=False):
        """
        Calculates the step height, width, and miscut angle using linear fitting.

        Args:
            x (numpy.ndarray): The x-axis data.
            z (numpy.ndarray): The z-axis data (height).
            peak_indices (numpy.ndarray): The indices of the peaks.
            valley_indices (numpy.ndarray): The indices of the valleys.
            fixed_height (float): The fixed step height.
            demo (bool, optional): If True, prints the calculated step properties. Defaults to False.

        Returns:
            tuple: step_heights, step_widths, miscut angles.
        """
        step_widths = []
        for i, v_ind in enumerate(valley_indices):
            x_valley, z_valley = x[v_ind], z[v_ind]

            # Ignore if there's no peak on the left
            if x_valley < np.min(x[peak_indices]): continue
            # If there's no peak on the right, then the valley is the last one
            if x_valley > np.max(x[peak_indices]): continue

            # Find the nearest peak on the left of the valley v_ind
            peaks_lhs = peak_indices[np.where(x[peak_indices] < x_valley)]
            left_peak_indice = peaks_lhs[np.argmax(peaks_lhs)]
            x_left_peak, z_left_peak = x[left_peak_indice], z[left_peak_indice]

            # Find the nearest peak on the right of the valley v_ind
            peaks_rhs = peak_indices[np.where(x[peak_indices] > x_valley)]
            right_peak_indice = peaks_rhs[np.argmin(peaks_rhs)]
            x_right_peak, z_right_peak = x[right_peak_indice], z[right_peak_indice]

            # Ignore if can't make a peak, valley, peak sequence
            if i!=0 and i!=len(valley_indices)-1:
                if  x[valley_indices[i-1]] > x_left_peak or x[valley_indices[i+1]] < x_right_peak:
                    continue
            
            # Fit the linear function between the right peak and the valley
            m, b = scipy.stats.linregress(x=[x_right_peak, x_valley], y=[z_right_peak, z_valley])[0:2]
            m = (z_right_peak-z_valley)/(x_right_peak-x_valley)
            b = z_valley - m*x_valley

            # Calculate the euclidean distance between the left peak and fitted linear function
            step_width = np.abs((m * x_left_peak - z_left_peak + b)) / (np.sqrt(m**2 + 1))
            step_widths.append(step_width)
            
            if demo:
                print(f'step {i}: step_width: {step_width:.2e}, left_peak: ({x_left_peak:.2e}, {z_left_peak:.2e}), valley: ({x_valley:.2e}, {z_valley:.2e}), right_peak: ({x_right_peak:.2e}, {z_right_peak:.2e})')
                
        step_heights = np.full(len(step_widths), fixed_height)
        miscut = np.degrees(np.arctan(step_heights/step_widths))
        
        if demo:
            print('Results:')
            print(f"  Average step height = {np.mean(step_heights):.2e}, Standard deviation = {np.std(step_heights):.2e}")
            print(f"  Average step width = {np.mean(step_widths):.2e}, Standard deviation = {np.std(step_widths):.2e}")
            print(f"  Average miscut = {np.mean(miscut):.3f}°, Standard deviation = {np.std(miscut):.3f}°")
        return step_heights, step_widths, miscut

    def clean_data(self, step_heights, step_widths, miscut, std_range=1, demo=False):
        """
        Cleans the data by removing outliers based on the specified standard deviation range.

        Args:
            step_heights (numpy.ndarray): The heights of the steps.
            step_widths (numpy.ndarray): The widths of the steps.
            miscut (numpy.ndarray): The miscut angles of the steps.
            std_range (float, optional): The range of standard deviation to remove outliers. Defaults to 1.
            demo (bool, optional): If True, prints the cleaned results. Defaults to False.

        Returns:
            tuple: Cleaned step heights, step widths, and miscut angles.
        """
        # Remove outliers
        miscut = miscut[np.abs(miscut-np.mean(miscut))<std_range*np.std(miscut)]
        step_heights = step_heights[np.abs(step_heights-np.mean(step_heights))<std_range*np.std(step_heights)]
        step_widths = step_widths[np.abs(step_widths-np.mean(step_widths))<std_range*np.std(step_widths)]
        if demo:
            print('Cleaned results:')
            print(f"  Average step height = {np.mean(step_heights):.2e}, Standard deviation = {np.std(step_heights):.2e}")
            print(f"  Average step width = {np.mean(step_widths):.2e}, Standard deviation = {np.std(step_widths):.2e}")
            print(f"  Average miscut = {np.mean(miscut):.3f}°, Standard deviation = {np.std(miscut):.3f}°")
        return step_heights, step_widths, miscut

    def calculate_substrate_properties(self, image_rot, size_rot, xz_angle, prominence=1e-11, width=2, style='simple', fixed_height=None, std_range=1, demo=False):
        """
        Calculates the properties of the substrate including step height, width, and miscut.

        Args:
            image_rot (numpy.ndarray): The rotated image.
            size_rot (float): The size of the rotated image in meters.
            prominence (float, optional): The prominence of peaks. Defaults to 1e-11.
            width (float, optional): The width of peaks. Defaults to 2.
            fixed_height (float, optional): The height of the step, provided if available from literature. Defaults to None.
            std_range (float, optional): The range of standard deviation to remove outliers. Defaults to 1.
            style (str, optional): The style of calculation ('simple' or 'fit'). Defaults to 'simple'.
            demo (bool, optional): If True, prints the calculated substrate properties. Defaults to False.

        Returns:
            dict: Dictionary containing step heights, step widths, and miscut angles.
        """
        step_heights_list, step_widths_list, miscut_list = [], [], []
        for j in range(self.pixels//4, self.pixels*3//4, 10):
            x, z, peak_indices, valley_indices = self.slice_rotate(image_rot, size_rot, j, prominence, width, xz_angle=xz_angle, demo=demo)
            
            if style == 'simple':
                step_heights, step_widths, miscut = self.calculate_simple(x, z, peak_indices, fixed_height=fixed_height, demo=demo)
            elif style == 'fit':
                step_heights, step_widths, miscut = self.calculate_fit(x, z, peak_indices, valley_indices, fixed_height=fixed_height, demo=demo)  
            step_heights_list.append(step_heights)
            step_widths_list.append(step_widths)
            miscut_list.append(miscut)
        
        step_heights = np.concatenate(step_heights_list)
        step_widths = np.concatenate(step_widths_list)
        miscut = np.concatenate(miscut_list)
        substrate_properties = {'step_heights': step_heights, 'step_widths': step_widths, 'miscut': miscut}

        print(f"Step height = {np.mean(step_heights):.2e} +- {np.std(step_heights):.2e}")
        print(f"Step width = {np.mean(step_widths):.2e} +- {np.std(step_widths):.2e}")
        print(f"Miscut = {np.mean(miscut):.3f}° +- {np.std(miscut):.3f}°")
        return substrate_properties

def visualize_afm_image(img, colorbar_range, figsize=(6,4), scalebar_dict=None, filename=None, printing=None, **kwargs):
    """
    Visualizes AFM image with scalebar and colorbar.

    Args:
        img (numpy.ndarray): AFM image as a 2D numpy array.
        colorbar_range (tuple): Range of the colorbar.
        figsize (tuple, optional): Size of the figure. Defaults to (6,4).
        scalebar_dict (dict, optional): Dictionary of scalebar parameters. Defaults to None.
        filename (str, optional): Filename to save the image. Defaults to None.
        printing (matplotlib.backends.backend_pdf.PdfPages, optional): PdfPages object for saving. Defaults to None.
        **kwargs: Additional keyword arguments for saving the image.

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(img)
    
    if scalebar_dict:
        scalebar(ax, image_size=scalebar_dict['image_size'], scale_size=scalebar_dict['scale_size'], 
                 units=scalebar_dict['units'], loc='br')

    if isinstance(colorbar_range, tuple) or isinstance(colorbar_range, list):
        im.set_clim(colorbar_range) 
        
    fig.colorbar(im, ax=ax)
    
    ax.tick_params(which='both', bottom=False, left=False, right=False, top=False, labelbottom=False)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    # Prints the figure
    if printing is not None and filename is not None:
        printing.savefig(fig, filename, **kwargs)  
    plt.show()

Path = path.Path
PathPatch = patches.PathPatch
erf = special.erf
cmap = plt.get_cmap('viridis')

def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds a path to the figure.

    Args:
        axes (matplotlib.axes.Axes): The axes to which the plot is added.
        locations (numpy.ndarray): The location to position the path.
        facecolor (str): The facecolor of the path.
        edgecolor (str): The edgecolor of the path.
        linestyle (str): The style of the line, using conventional matplotlib styles.
        lineweight (float): The thickness of the line.

    Returns:
        None
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    # Extracts the vertices used to construct the path
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    # Makes a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          ls=linestyle, lw=lineweight)
    # Adds path to axes
    axes.add_patch(pathpatch)

def scalebar(axes, image_size, scale_size, units='nm', loc='br'):
    """
    Adds a scalebar to figures.

    Args:
        axes (matplotlib.axes.Axes): The axes to which the plot is added.
        image_size (int): The size of the image in nm.
        scale_size (str): The size of the scalebar in units of nm.
        units (str, optional): The units for the label. Defaults to 'nm'.
        loc (str, optional): The location of the label. Defaults to 'br'.

    Returns:
        None
    """

    # Gets the size of the image
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(
        np.int32(np.floor(x_lim[1] - x_lim[0]))), np.abs(np.int32(np.floor(y_lim[1] - y_lim[0])))
    # Computes the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1],
                          np.int32(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1],
                          np.int32(np.floor(image_size)))

    # Sets the location of the scalebar
    if loc == 'br':
        x_start = x_point[np.int32(.9 * image_size // 1)]
        x_end = x_point[np.int32((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int32(.1 * image_size // 1)]
        y_end = y_point[np.int32((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int32((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int32(.9 * image_size // 1)]
        x_end = x_point[np.int32((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int32(.9 * image_size // 1)]
        y_end = y_point[np.int32((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int32((.9 - .075) * image_size // 1)]

    # Makes the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    # Adds the text label for the scalebar
    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])

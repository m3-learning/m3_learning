import numpy as np
import imutils
from matplotlib import (pyplot as plt, animation, colors, ticker, path, patches, patheffects)
import plotly.graph_objects as go
import pylab as pl
import scipy
from scipy import special
from scipy import signal
from scipy.signal import savgol_filter

class afm_substrate():
    """
    This class is designed to facilitate the analysis of an atomic force microscopy (AFM) substrate image. 
    The class includes methods for image rotation, coordinate transformation, peak detection, and step parameter calculation.
    """ 
    def __init__(self, img, pixels, size):
        '''
        img: the image to be analyzed
        pixels: the number of pixels in the image
        size: the size of the image in meters
        '''
        self.img = img
        self.pixels = pixels
        self.size = size
    
    def rotate_image(self, angle, colorbar_range=None, demo=True):
        '''
        angle: the angle to rotate the image in degrees
        '''
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
        '''
        x: the x coordinates of the image
        z: the z coordinates of the image
        xz_angle: the angle to rotate the xz plane
        '''
        theta = np.radians(xz_angle)
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        return x_rot, z_rot

    def show_peaks(self, x, z, peaks=None, valleys=None):
        '''
        x: the x-axis data
        z: the z-axis data - height
        peaks: the indices of the peaks
        valleys: the indices of the valleys
        '''
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
        '''
        img_rot: the rotated image
        size: the size of the image in meters
        j: the column to slice
        xz_angle: the angle between the x and z axes in degrees
        '''
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

            # rotate the xz plane to level the step
            x_norm_rot, z_norm_rot = self.rotate_xz(x_norm, z_norm, xz_angle)
            x, z = x_norm_rot * (x_max - x_min) + x_min, z_norm_rot * (z_max - z_min) + z_min
        
        if demo:
            self.show_peaks(x, z, peak_indices, valley_indices)
        return x, z, peak_indices, valley_indices


    def calculate_simple(self, x, z, peak_indices, fixed_height=None, demo=False):
        '''
        Calculate the height, width, and miscut of the steps in a straight forward way.
        Calculate the height and width of each step from the rotated line profile.
        x: the x-axis data
        z: the z-axis data - height
        peak_indices: the indices of the peaks
        fixed_height: the height of the steps
        '''

        # find the level of z and step height and width
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
        '''
        calculate the step height, width and miscut angle. 
        The step height is calculated by the perpendicular distance between lower step bottom point (valley) and the fitting function of higher step edge (line between left peak and right peak). 
        x: the x-axis data
        z: the z-axis data - height
        peak_indices: the indices of the peaks
        valley_indices: the indices of the valleys
        fixed_height: the fixed step height
        demo: whether to show the demo plot
        '''
        # print(valley_indices)
        step_widths = []
        for i, v_ind in enumerate(valley_indices):
            x_valley, z_valley = x[v_ind], z[v_ind]

            # ignore if there's no peak on the left
            if x_valley < np.min(x[peak_indices]): continue
            # if there's no peak on the right, then the valley is the last one
            if x_valley > np.max(x[peak_indices]): continue

            # find the nearest peak on the left of the valley v_ind
            peaks_lhs = peak_indices[np.where(x[peak_indices] < x_valley)]
            left_peak_indice = peaks_lhs[np.argmax(peaks_lhs)]
            x_left_peak, z_left_peak = x[left_peak_indice], z[left_peak_indice]

            # find the nearest peak on the right of the valley v_ind
            peaks_rhs = peak_indices[np.where(x[peak_indices] > x_valley)]
            right_peak_indice = peaks_rhs[np.argmin(peaks_rhs)]
            x_right_peak, z_right_peak = x[right_peak_indice], z[right_peak_indice]

            # ignore if can't make a peak, valley, peak sequence
            if i!=0 and i!=len(valley_indices)-1:
                if  x[valley_indices[i-1]] > x_left_peak or x[valley_indices[i+1]] < x_right_peak:
                    continue
            
            # fit the linear function between the right peak and the valley
            m, b = scipy.stats.linregress(x=[x_right_peak, x_valley], y=[z_right_peak, z_valley])[0:2]
            m = (z_right_peak-z_valley)/(x_right_peak-x_valley)
            b = z_valley - m*x_valley

            # calculate the euclidean distance between the left peak and fitted linear function
            step_width = np.abs((m * x_left_peak - z_left_peak + b)) / (np.sqrt(m**2 + 1))
            step_widths.append(step_width)
            
            # print left peak, valley, right peak
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
        '''
        step_heights: the heights of the steps
        step_widths: the widths of the steps
        miscut: the miscut of the steps
        std_range: the range of standard deviation to remove outliers
        demo: whether to show the cleaned results
        '''
        # remove outliers
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
        '''
        image_rot: the rotated image
        size_rot: the size of the rotated image in meters
        prominence: the prominence of the peaks
        width: the width of the peaks
        fixed_height: the height of the step, provide if can be acquired from literature
        std_range: the range of standard deviation to remove outliers
        '''
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
    '''
    Visualize AFM image with scalebar and colorbar.
    -----------
    Parameters:
        img: 2D numpy array, AFM image;
        colorbar_range: tuple, Range of colorbar;
        scalebar_dict: dict, Dictionary of scalebar parameters;
        filename: str, Filename to save the image;
    -----------
    Returns: None;
    '''

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

    # prints the figure
    if printing is not None and filename is not None:
        printing.savefig(fig, filename, **kwargs)  
    plt.show()


"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""
from scipy import special
from matplotlib import animation, colors, ticker, path, patches, patheffects

Path = path.Path
PathPatch = patches.PathPatch
erf = special.erf
cmap = plt.get_cmap('viridis')

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
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    #  makes a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          ls=linestyle, lw=lineweight)
    # adds path to axes
    axes.add_patch(pathpatch)

def scalebar(axes, image_size, scale_size, units='nm', loc='br'):
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
    x_size, y_size = np.abs(
        np.int32(np.floor(x_lim[1] - x_lim[0]))), np.abs(np.int32(np.floor(y_lim[1] - y_lim[0])))
    # computes the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1],
                          np.int32(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1],
                          np.int32(np.floor(image_size)))

    # sets the location of the scalebar"
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

    # makes the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    # adds the text label for the scalebar
    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])
    
    
# pip install xrayutilities
import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xu
from matplotlib.ticker import MultipleLocator
import glob
from matplotlib import ticker, cm, colors


def plot_xrd(ax, files, labels, title=None, xrange=(0,90), diff=1e3, pad_sequence=[]):
    """
    Plot X-ray diffraction patterns.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot the patterns.
    - files (list of str): The paths to the files containing the X-ray diffraction data.
    - labels (list of str): The labels for each X-ray diffraction pattern.
    - title (str, optional): The title for the plot (default: None).
    - xrange (tuple, optional): The x-axis range of the plot (default: (0, 90)).
    - diff (float, optional): Scaling factor for intensity differences between patterns (default: 1e3).
    - pad_sequence (list, optional): Padding sequence for X-ray diffraction patterns with different scan ranges (default: []).

    Returns:
    None
    """
    
    Xs, Ys = [], []
    length_list = []
    for file in files:
        out = xu.io.getxrdml_scan(file)
        Xs.append(out[0])
        Ys.append(out[1])
        length_list.append(len(out[0]))
        
    if np.mean(length_list) != np.max(length_list):
        if pad_sequence == []:
            print('Different scan ranges, input pad_sequence to pad')
            return 
        else:
            for i in range(len(Ys)):
                Ys[i] = np.pad(Ys[i], pad_sequence[i], mode='median')
    X = Xs[np.argmax(length_list)]
        
    # fig, axes = plt.subplots(figsize=figsize)
    
    for i, Y in enumerate(Ys):
        Y[Y==0] = 1  # remove all 0 value
        if diff:
            Y = Y * diff**(len(Ys)-i-1)
        ax.plot(X, Y, label=labels[i])
        
    ax.set_xlabel(r"2$\Theta}$", )
    ax.set_ylabel('Intensity [a.u.]')
    ax.legend()

    ax.set_yscale('log',base=10) 
    ax.set_xlim(xrange)
    if title: ax.set_title(title)
    # ax.set_xticks(np.arange(*xrange, 1))


def plot_rsm(ax, file, reciprocal_space=True, title=None):
    """
    Plot a reciprocal space map or a real space map.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot the map.
    - file (str): The path to the file containing the data.
    - reciprocal_space (bool, optional): Whether to plot a reciprocal space map (default: True).
    - title (str, optional): The title for the plot (default: None).

    Returns:
    None
    """
    
    curve_shape = xu.io.getxrdml_scan(file)[0].shape
    omega, two_theta, intensity = xu.io.panalytical_xml.getxrdml_map(file)

    omega = omega.reshape(curve_shape)
    two_theta = two_theta.reshape(curve_shape)
    intensity = intensity.reshape(curve_shape)
    intensity[intensity==0]=1

    if reciprocal_space:
        wavelength = 1.54 # unit: angstrom
        Qz = (1/wavelength)*(np.sin(np.deg2rad(two_theta-omega))+np.sin(np.deg2rad(omega))) 
        Qx = (1/wavelength)*(np.cos(np.deg2rad(omega))-np.cos(np.deg2rad(two_theta-omega)))
        cs = ax.contourf(Qx, Qz, intensity, locator=ticker.LogLocator(), cmap=cm.viridis, norm=colors.LogNorm())
    else:
        cs = ax.contourf(omega, two_theta, intensity, locator=ticker.LogLocator(), cmap=cm.viridis, norm=colors.LogNorm())

    formatter = ticker.LogFormatterMathtext(base=10, labelOnlyBase=False)
    ax.set_xlabel(r"Qx r'$\AA$'")
    ax.set_ylabel(r"Qz r'$\AA$'")

    plt.colorbar(cs, ax=ax, format=formatter)
    if title: ax.set_title(title)
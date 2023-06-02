import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import seaborn as sns
from scipy.signal import savgol_filter
from m3_learning.viz.layout import layout_fig, labelfigs


class Viz:
    def __init__(self, printing = None):
        """
        Initialize a Viz object.

        Args:
            printing (optional): An object used for saving figures. Defaults to None.
        """
        self.Printer = printing

    @staticmethod
    def make_fine_step(x, transparency, step, color, saturation=1, savgol_filter_level=(15,1)):
        """
        Create a fine step for given data.

        Args:
            x (ndarray): Input values.
            transparency (ndarray): Transparency values.
            step (int): Number of steps.
            color: The color to use.
            saturation (float, optional): Saturation level. Defaults to 1.
            savgol_filter_level (tuple, optional): Savitzky-Golay filter level. Defaults to (15, 1).

        Returns:
            tuple: Tuple containing the fine step of x and the colors.
        """
        x_FineStep = np.hstack([np.linspace(start, stop, num=step+1, endpoint=True)[:-1] for start, stop in zip(x, x[1:])])
        
        transparency_FineStep = np.hstack([np.linspace(start, stop, num=step+1, endpoint=True)[:-1] for start, stop in zip(transparency, transparency[1:])])
        if not isinstance(savgol_filter_level, type(None)):
            transparency_FineStep_before = np.copy(transparency_FineStep)
            transparency_FineStep = savgol_filter(transparency_FineStep, savgol_filter_level[0]*step+1, savgol_filter_level[1])

        transparency_FineStep_norm = np.expand_dims((transparency_FineStep / max(transparency_FineStep)) * saturation, 1)
        transparency_FineStep_norm[transparency_FineStep_norm<0] = 0
        
        colors = np.repeat([[*color]], len(transparency_FineStep_norm), 0)
        colors_all = np.concatenate([colors, transparency_FineStep_norm], 1)
        return x_FineStep, colors_all


    @staticmethod
    def two_color_array(x_all, x1, x2, c1, c2, transparency=1):
        """
        Create a two-color array based on given conditions.

        Args:
            x_all (ndarray): All values.
            x1 (ndarray): Values to be colored with c1.
            x2 (ndarray): Values to be colored with c2.
            c1: Color 1.
            c2: Color 2.
            transparency (float, optional): Transparency of the colors. Defaults to 1.

        Returns:
            ndarray: Color array.
        """

        color_array = np.zeros([len(x_all), 4], dtype=np.float32)
        if len(c1) < 4:
            color_array[np.isin(x_all, x1)] = [*c1, transparency]
        else:
            color_array[np.isin(x_all, x1)] = c1

        if len(c2) < 4:
            color_array[np.isin(x_all, x2)] = [*c2, transparency]
        else:
            color_array[np.isin(x_all, x2)] = c2
        return color_array


    @staticmethod
    def draw_background_colors(ax, bg_colors):
        """
        Draw background colors on the given axes.

        Args:
            ax: Axes object.
            bg_colors: Background colors.

        Returns:
            None
        """
        if isinstance(bg_colors, tuple):
            ax.set_facecolor(bg_colors)
        elif bg_colors is not None:
            x_coor = bg_colors[:, 0]
            colors = bg_colors[:, 1:]
            for i in range(len(x_coor)):
                if i == 0:
                    end = (x_coor[i] + x_coor[i+1]) / 2
                    start = end - (x_coor[i+1] - x_coor[i])
                elif i == len(x_coor) - 1:
                    start = (x_coor[i-1] + x_coor[i]) / 2
                    end = start + (x_coor[i] - x_coor[i-1])
                else:
                    start = (x_coor[i-1] + x_coor[i]) / 2
                    end = (x_coor[i] + x_coor[i+1]) / 2
                ax.axvspan(start, end, facecolor=colors[i])
                
    @staticmethod
    def draw_boxes(ax, boxes, box_color):
        """
        Draw boxes on the given axes.

        Args:
            ax: Axes object.
            boxes: List of box coordinates.
            box_color: Color of the boxes.

        Returns:
            None
        """
        for (box_start, box_end) in boxes:
            ax.axvspan(box_start, box_end, facecolor=box_color, edgecolor=box_color)

    @staticmethod
    def find_nearest(array, value):
        """
        Find the nearest value in the array to the given value.

        Args:
            array: Input array.
            value: Target value.

        Returns:
            float: Nearest value in the array.
        """
        idx = np.abs(array - value).argmin()
        return array[idx]

    @staticmethod
    def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', 
                logscale=False, legend=None, ticks_both_sides=True):
        """
        Set labels and other properties of the given axes.

        Args:
            ax: Axes object.
            xlabel (str, optional): X-axis label. Defaults to None.
            ylabel (str, optional): Y-axis label. Defaults to None.
            title (str, optional): Plot title. Defaults to None.
            xlim (tuple, optional): X-axis limits. Defaults to None.
            ylim (tuple, optional): Y-axis limits. Defaults to None.
            yaxis_style (str, optional): Y-axis style. Defaults to 'sci'.
            logscale (bool, optional): Use log scale on the y-axis. Defaults to False.
            legend (list, optional): Legend labels. Defaults to None.
            ticks_both_sides (bool, optional): Display ticks on both sides of the axes. Defaults to True.

        Returns:
            None
        """
        if type(xlabel) != type(None): ax.set_xlabel(xlabel)
        if type(ylabel) != type(None): ax.set_ylabel(ylabel)
        if type(title) != type(None): ax.set_title(title)
        if type(xlim) != type(None): ax.set_xlim(xlim)
        if type(ylim) != type(None): ax.set_ylim(ylim)
        if yaxis_style == 'sci':
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
        if logscale: ax.set_yscale("log") 
        if legend: ax.legend(legend)
        ax.tick_params(axis="x",direction="in")
        ax.tick_params(axis="y",direction="in")
        if ticks_both_sides:
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')

    @staticmethod
    def plot_image_with_colorbar(fig, ax, image, style='3_values'):
        """
        Plot an image with a colorbar.

        Args:
            fig: Figure object.
            ax: Axes object.
            image: Image data.
            style (str, optional): Style of the colorbar ('3_values' or 'continuous'). Defaults to '3_values'.

        Returns:
            None
        """        
        im = ax.imshow(image, vmin=image.min(), vmax=image.max())
        cbar = plt.colorbar(im, ticks=[image.min(), image.max(), image.mean()])
        cbar.ax.set_yticklabels([image.min(), image.max(), image.mean()])

    @staticmethod
    def label_curves(ax, curve_x, curve_y, labels_dict):
        """
        Label curves on a plot.

        Args:
            ax: Axes object.
            curve_x: X-axis values of the curve.
            curve_y: Y-axis values of the curve.
            labels_dict: Dictionary of labels for specific x-values.

        Returns:
            None
        """
        if type(labels_dict) != type(None):
            for x in labels_dict.keys():
                y = curve_y[np.where(curve_x==Viz.find_nearest(curve_x, x))]
                pl.text(x, y, str(labels_dict[x]), color="g", fontsize=6)
                
                
    @staticmethod
    def plot_curve(ax, curve_x, curve_y, curve_x_fit=None, curve_y_fit=None, plot_colors=['k', 'r'], plot_type='scatter', 
                   markersize=1, xlabel=None, ylabel=None, xlim=None, ylim=None, logscale=False, yaxis_style='sci',
                   title=None, legend=None):
        """
        Plot a curve on the given axes.

        Args:
            ax: Axes object.
            curve_x: X-axis values of the curve.
            curve_y: Y-axis values of the curve.
            curve_x_fit (optional): X-axis values of the fitted curve. Defaults to None.
            curve_y_fit (optional): Y-axis values of the fitted curve. Defaults to None.
            plot_colors (list, optional): Colors for plotting the curve and the fitted curve. Defaults to ['k', 'r'].
            plot_type (str, optional): Type of plot to use ('scatter' or 'lineplot'). Defaults to 'scatter'.
            markersize (int, optional): Size of markers for scatter plot. Defaults to 1.
            xlabel (str, optional): X-axis label. Defaults to None.
            ylabel (str, optional): Y-axis label. Defaults to None.
            xlim (tuple, optional): X-axis limits. Defaults to None.
            ylim (tuple, optional): Y-axis limits. Defaults to None.
            logscale (bool, optional): Use log scale on the y-axis. Defaults to False.
            yaxis_style (str, optional): Y-axis style. Defaults to 'sci'.
            title (str, optional): Plot title. Defaults to None.
            legend (list, optional): Legend labels. Defaults to None.

        Returns:
            None
        """

        if plot_type == 'scatter':
            ax.plot(curve_x, curve_y, color=plot_colors[0], markersize=markersize)
            if not isinstance(curve_y_fit, type(None)):
                if not isinstance(curve_x_fit, type(None)):
                    ax.scatter(curve_x_fit, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_scatter(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
                else:
                    ax.scatter(curve_x, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_scatter(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                    
        if plot_type == 'lineplot':
            ax.plot(curve_x, curve_y, color=plot_colors[0], markersize=markersize)
            if not isinstance(curve_y_fit, type(None)):
                if not isinstance(curve_x_fit, type(None)):
                    ax.plot(curve_x_fit, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_lineplot(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
                else:
                    ax.plot(curve_x, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_lineplot(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                    
        Viz.set_labels(ax, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, yaxis_style=yaxis_style, 
                   logscale=logscale, legend=legend)
        
        
    @staticmethod
    def set_index(axes, index, total_length):
        """
        Set the index for subplots.

        Args:
            axes: Axes object.
            index (int): Index value.
            total_length (int): Total length of subplots.

        Returns:
            None
        """
        rows, img_per_row = axes.shape    
        if total_length <= img_per_row:
            index = index%img_per_row
        else:
            index = (index//img_per_row), index%img_per_row

    # fig, axes = plt.subplots(len(ys)//img_per_row+1*int(len(ys)%img_per_row>0), img_per_row, 
    #                          figsize=(16, subplot_height*len(ys)//img_per_row+1))  
    # def show_grid_plots(xs, ys, labels=None, ys_fit1=None, ys_fit2=None, img_per_row=4, subplot_height=3, ylim=None, legend=None):
    @staticmethod
    def show_grid_plots(axes, xs, ys, labels=None, xlabel=None, ylabel=None, ylim=None, legend=None, color=None):
        """
        Show a grid of plots.

        Args:
            axes: Axes object.
            xs: X-axis values.
            ys: Y-axis values.
            labels (optional): Labels for the plots. Defaults to None.
            xlabel (str, optional): X-axis label. Defaults to None.
            ylabel (str, optional): Y-axis label. Defaults to None.
            ylim (tuple, optional): Y-axis limits. Defaults to None.
            legend (list, optional): Legend labels. Defaults to None.
            color (str, optional): Color for the plots. Defaults to None.

        Returns:
            None
        """
        if type(labels) == type(None): labels = range(len(ys))
        if isinstance(color, type(None)): color = 'k'
        for i in range(len(ys)):
            # i = Viz.set_index(axes, i, total_length=len(ys))
            axes[i].plot(xs[i], ys[i], marker='.', markersize=2, color=color)
            Viz.set_labels(axes[i], xlabel=xlabel, ylabel=ylabel, ylim=ylim, legend=legend)
        if not isinstance(labels, type(None)):
            labelfigs(axes[i], 1, string_add=labels[i], loc='bm', size=6)
        # plt.show()
        
    @staticmethod
    def plot_loss_difference(ax1, x_all, y_all, x_coor_all, loss_diff, color_array, color_2, title=None):
        """
        Plot the loss difference.

        Args:
            ax1: Axes object.
            x_all: X-axis values.
            y_all: Y-axis values.
            x_coor_all: X-axis values for the loss difference.
            loss_diff: Loss difference values.
            color_array: Array of colors for the background.
            color_2: Color for the loss difference plot.
            title (str, optional): Plot title. Defaults to None.

        Returns:
            None
        """
        Viz.draw_background_colors(ax1, color_array)
        ax1.scatter(x_all, y_all, c='k', s=1)
        Viz.set_labels(ax1, xlabel='Time (s)', ylabel='Intensity (a.u.)')
        ax1.tick_params(axis="y", labelcolor='k')
        ax1.set_ylabel('Intensity (a.u.)', color='k')
        ax1.tick_params(axis="x",direction="in")
    
        ax2 = ax1.twinx()
        ax2.scatter(x_coor_all, loss_diff, color=color_2, s=1)
        Viz.set_labels(ax2, xlabel='Time (s)', ylabel='Loss difference (a.u.)', logscale=True)
        ax2.tick_params(axis="y", color=color_2, labelcolor=color_2)
        ax2.set_ylabel('Loss difference (a.u.)', color=color_2)
        ax2.tick_params(axis="x",direction="in")
        plt.title(title)


    @staticmethod
    def plot_fit_details(x, y1, y2, y3, index_list, save_name=None, printing=None):
        """
        Plot the fit details.

        Args:
            x: X-axis values.
            y1: Y-axis values for raw data.
            y2: Y-axis values for prediction.
            y3: Y-axis values for failed data.
            index_list: List of index values.
            save_name (str, optional): Name to save the plot. Defaults to None.
            printing: Printing object.

        Returns:
            None
        """
        
        mod = 6
        if len(y1)//mod > 10:
            n_page = len(y1) // mod // 10 + 1
            for np in range(n_page):
                start_plot = np*10*mod
                if np == n_page-1:
                    n_plot = len(y1)%(10*mod) + 1
                else:
                    n_plot = 10*mod

                fig, axes = layout_fig(n_plot, mod=mod, figsize=(6, 1*(n_plot//mod+1)))
                axes = axes.flatten()[:n_plot]
                for i in range(start_plot, start_plot+n_plot):
                    if np == n_page-1 and i == start_plot+n_plot-1:                    
                        handles, labels = axes[-2].get_legend_handles_labels()
                        axes[-1].legend(handles=handles, labels=labels, loc='center')
                        axes[-1].set_xticks([])
                        axes[-1].set_yticks([])
                        axes[-1].set_frame_on(False)

                    else:
                        xlabel, ylabel = None, None
                        l1 = axes[i%(10*mod)].plot(x[i], y1[i], marker='.', markersize=2, 
                                        color=(44/255,123/255,182/255, 0.5), label='Raw data')
                        l2 = axes[i%(10*mod)].plot(x[i], y2[i], linewidth=2, label='Prediction')
                        l3 = axes[i%(10*mod)].plot(x[i], y3[i], linewidth=1, label='Failed')

                        if (i%(10*mod)+1) % mod == 1: ylabel = 'Intensity (a.u.)'
                        if np == n_page-1 and i%(10*mod)+1 >= len(axes)-mod: xlabel = 'Time (s)'

                        Viz.set_labels(axes[i%(10*mod)], xlabel=xlabel, ylabel=ylabel)

                        labelfigs(axes[i%(10*mod)], None, string_add=str(index_list[i]), loc='ct', size=8, style='b')
                        # labelfigs(axes[i%(10*mod)], None, string_add=str(index_list[i]), loc='ct', style='b')
                        axes[i%(10*mod)].set_xticks([])
                        axes[i%(10*mod)].set_yticks([])
                        axes[i%(10*mod)].xaxis.set_tick_params(labelbottom=False)
                        axes[i%(10*mod)].yaxis.set_tick_params(labelleft=False)

                plt.tight_layout(pad=-0.5, w_pad=-1, h_pad=-0.5)
                if save_name:
                    printing.savefig(fig, save_name+'-'+str(np+1))
                plt.show()
        else:
            fig, axes = layout_fig(len(y1)+1, mod=mod, figsize=(6, 1*len(y1)//mod))
            axes = axes.flatten()[:len(y1)+1]
            for i in range(len(x)):
                xlabel='Time (s)'
                ylabel='Intensity (a.u.)'

                l1 = axes[i].plot(x[i], y1[i], marker='.', markersize=2, 
                                color=(44/255,123/255,182/255, 0.5), label='Raw data')
                l2 = axes[i].plot(x[i], y2[i], linewidth=2, label='Prediction')
                l3 = axes[i].plot(x[i], y3[i], linewidth=1, label='Failed')
                if i+1 < len(axes)-mod: xlabel = None
                if not (i+1) % mod == 1: ylabel = None
                Viz.set_labels(axes[i], xlabel=xlabel, ylabel=ylabel)
                labelfigs(axes[i], None, string_add=str(index_list[i]), loc='ct', size=8, style='b')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].xaxis.set_tick_params(labelbottom=False)
                axes[i].yaxis.set_tick_params(labelleft=False)

            handles, labels = axes[-2].get_legend_handles_labels()
            axes[-1].legend(handles=handles, labels=labels, loc='center')
            axes[-1].set_xticks([])
            axes[-1].set_yticks([])
            axes[-1].set_frame_on(False)

            plt.tight_layout(pad=-0.5, w_pad=-1, h_pad=-0.5)
            if save_name:
                printing.savefig(fig, save_name)
            plt.show()

    @staticmethod
    def label_violinplot(ax, data, label_type='average', text_pos='center'):
        """
        Label a violin plot.

        Args:
            ax: Axes object.
            data: Data for the violin plot.
            label_type (str, optional): Type of label to use ('average' or 'number'). Defaults to 'average'.
            text_pos (str, optional): Position of the label text ('center' or 'right'). Defaults to 'center'.

        Returns:
            None
        """
        
        # Calculate number of obs per group & median to position labels
        xloc = range(len(data))
        yloc, text = [], [] 
        
        for i, d in enumerate(data):
            yloc.append(np.median(d))
            
            if label_type == 'number':
                text.append("n: "+str(len(d)))
            
            if label_type == 'average':
                text.append(str(round(np.median(d), 4)))

        for tick, label in zip(xloc, ax.get_xticklabels()):
            if text_pos == 'center':
                ax.text(xloc[tick], yloc[tick]*1.1, text[tick], horizontalalignment='center', size=14, weight='semibold')
            if text_pos == 'right':
                ax.text(xloc[tick]+0.02, yloc[tick]*0.7, text[tick], horizontalalignment='left', size=14, weight='semibold')
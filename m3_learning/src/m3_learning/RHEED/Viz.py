import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import seaborn as sns
from scipy.signal import savgol_filter
from m3_learning.viz.layout import imagemap, layout_fig, labelfigs


def set_style_RHEED(name="default"):
    """Function to implement custom default style for graphs

    Args:
        name (str, optional): style name. Defaults to "default".
    """
    if name == "RHEED_plot":
        try:
            rc_plot = {'figure.figsize':(12,2.5),
                    'axes.facecolor':'white',
                    'axes.grid': False,
                    'axes.titlesize': 18,
                    'axes.labelsize': 18,

                    'xtick.labelsize': 14,
                    'xtick.direction': 'in',
                    'xtick.top': True,
                    'xtick.bottom': True,
                    'xtick.labelbottom': True,
                    'xtick.labeltop': False,
                    
                    'ytick.labelsize': 14,
                    'ytick.direction': 'in',
                    'ytick.right': True,
                    'ytick.left': True,
                    'ytick.labelleft': True,
                    'ytick.labelright': False,

                    'legend.fontsize': 10,
                    'font.family': 'sans-serif'}

            plt.rcParams.update(rc_plot)

        except:
            pass

class Viz:
    def __init__(self, printing = None):
        self.Printer = printing

    @staticmethod
    def make_fine_step(x, transparency, step, color, saturation=1, savgol_filter_level=(15,1)):
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
        '''
        x_all: all values
        x1: values to be colored with c1
        x2: values to be colored with c2
        c1: color 1
        c2: color 2
        transparency: transparency of the colors
        '''
        color_array = np.zeros([len(x_all), 4], dtype=np.float32)
        color_array[np.isin(x_all, x1)] = [*c1, transparency]
        color_array[np.isin(x_all, x2)] = [*c2, transparency]
        return color_array


    @staticmethod
    def draw_background_colors(ax, bg_colors):
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
        for (box_start, box_end) in boxes:
            ax.axvspan(box_start, box_end, facecolor=box_color, edgecolor=box_color)

    @staticmethod
    def find_nearest(array, value):
        idx = np.abs(array - value).argmin()
        return array[idx]

    @staticmethod
    def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', 
                logscale=False, legend=None):
        if type(xlabel) != type(None): ax.set_xlabel(xlabel)
        if type(ylabel) != type(None): ax.set_ylabel(ylabel)
        if type(title) != type(None): ax.set_title(title)
        if type(xlim) != type(None): ax.set_xlim(xlim)
        if type(ylim) != type(None): ax.set_ylim(ylim)
        if yaxis_style == 'sci':
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
        if logscale: plt.yscale("log") 
        if legend: plt.legend(legend)


    @staticmethod
    def label_curves(ax, curve_x, curve_y, labels_dict):
        if type(labels_dict) != type(None):
            for x in labels_dict.keys():
                y = curve_y[np.where(curve_x==Viz.find_nearest(curve_x, x))]
                pl.text(x, y, str(labels_dict[x]), color="g", fontsize=6)
                
                
    @staticmethod
    def plot_curve(curve_x, curve_y, curve_x_fit=None, curve_y_fit=None, plot_colors=['k', 'r'], plot_type='scatter', 
                   markersize=1, xlabel=None, ylabel=None, xlim=None, ylim=None, logscale=False, yaxis_style='sci',
                   title=None, legend=None, figsize=(12,2.5), filename=None, printing=None, **kwargs):
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if plot_type == 'scatter':
            ax.plot(curve_x, curve_y, color=plot_colors[0], markersize=markersize)
            if not isinstance(curve_y_fit, type(None)):
                if not isinstance(curve_x_fit, type(None)):
                    plt.scatter(curve_x_fit, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_scatter(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
                else:
                    plt.scatter(curve_x, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_scatter(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                    
        if plot_type == 'lineplot':
            plt.plot(curve_x, curve_y, color=plot_colors[0], markersize=markersize)
            if not isinstance(curve_y_fit, type(None)):
                if not isinstance(curve_x_fit, type(None)):
                    plt.plot(curve_x_fit, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_lineplot(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
                else:
                    plt.plot(curve_x, curve_y_fit, color=plot_colors[1], markersize=markersize)
                    # plot_lineplot(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                    
        Viz.set_labels(ax, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, yaxis_style=yaxis_style, 
                   logscale=logscale, legend=legend)
        
        # prints the figure
        if printing is not None and filename is not None:
            printing.savefig(fig, filename, **kwargs)  
        plt.show()

    @staticmethod
    def set_index(axes, index, total_length):
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
    def plot_loss_difference(x_all, y_all, x_coor_all, loss_diff, color_array, color_blue, title=None):

        fig, ax1 = plt.subplots(1, 1, figsize=(8, 2))
        Viz.draw_background_colors(ax1, color_array)
        ax1.scatter(x_all, y_all, c='k', s=1)
        Viz.set_labels(ax1, xlabel='Time (s)', ylabel='Intensity (a.u.)')
        ax1.tick_params(axis="y", labelcolor='k')
        ax1.set_ylabel('Intensity (a.u.)', color='k')

        ax2 = ax1.twinx()
        ax2.scatter(x_coor_all, loss_diff, color=np.array(color_blue).reshape(1,-1), s=1)
        Viz.set_labels(ax2, xlabel='Time (s)', ylabel='Loss difference (a.u.)', logscale=True)
        ax2.tick_params(axis="y", color=color_blue, labelcolor=color_blue)
        ax2.set_ylabel('Loss difference (a.u.)', color=color_blue)
        plt.title(title)
        plt.show()


    @staticmethod
    def plot_fit_details(x, y1, y2, y3, index_list):
        fig, axes = layout_fig(len(y1)+1, mod=4, figsize=(8, 1.8*len(y1)//4))
        axes = axes.flatten()[:len(y1)+1]
        for i in range(len(x)):
            xlabel, ylabel='Time (s)', 'Intensity (a.u.)'

            l1 = axes[i].plot(x[i], y1[i], marker='.', markersize=2, 
                            color=(44/255,123/255,182/255, 0.5), label='Raw data')
            l2 = axes[i].plot(x[i], y2[i], linewidth=2, label='Prediction')
            l3 = axes[i].plot(x[i], y3[i], linewidth=1, label='Failed')
            if i+1 < len(axes)-4: xlabel = None
            if not (i+1) % 4 == 1: ylabel = None
            Viz.set_labels(axes[i], xlabel=xlabel, ylabel=ylabel)
            labelfigs(axes[i], 1, string_add=index_list[i], loc='bm', size=10)

        handles, labels = axes[-2].get_legend_handles_labels()
        axes[-1].legend(handles=handles, labels=labels, loc='center')
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        axes[-1].set_frame_on(False)

        plt.tight_layout(pad=-0.5, w_pad=-1, h_pad=-0.5)
        plt.show()

    @staticmethod
    def label_violinplot(ax, data, label_type='average', text_pos='center'):
        '''
        data: a list of list or numpy array for different 
        '''
        
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



# this should goes into the function in dataset's class
# class Viz:
#     def __init__(self, dataset, printing = None):
#         self.dataset = dataset
#         self.Printer = printing

#     def RHEED_spot(self, growth, index, figsize=None, filename = None, **kwargs):
        
#         # if "growth" in kwargs:
#         #     self.dataset.growth = kwargs["growth"]
#         if figsize is None: figsize = (1.5, 1.5)
#         fig, ax = plt.subplots(figsize = figsize)
#         data = self.dataset.growth_dataset(growth, index)
#         imagemap(ax, data, divider_=True)

#         if filename is True: 
#             filename = f"RHEED_{self.dataset.sample_name}_{growth}_{index}"
                
#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename, **kwargs)
#         plt.show()

#     def RHEED_parameter(self, growth, spot, index, figsize=None, filename = None, **kwargs):
#         if figsize is None:
#             figsize = (1.25*3, 1.25*1)
#         # "img_mean", "img_rec_sum", "img_rec_max", "img_rec_mean", "height", "x", "y", "width_x", "width_y".
#         img = self.dataset.growth_dataset(growth, spot, 'raw_image', index)
#         img_rec = self.dataset.growth_dataset(growth, spot, 'reconstructed_image', index)
#         img_sum = self.dataset.growth_dataset(growth, spot, 'img_sum', index)
#         img_max = self.dataset.growth_dataset(growth, spot, 'img_max', index)
#         img_mean = self.dataset.growth_dataset(growth, spot, 'img_mean', index)
#         img_rec_sum = self.dataset.growth_dataset(growth, spot, 'img_rec_sum', index)
#         img_rec_max = self.dataset.growth_dataset(growth, spot, 'img_rec_max', index)
#         img_rec_mean = self.dataset.growth_dataset(growth, spot, 'img_rec_mean', index)
#         height = self.dataset.growth_dataset(growth, spot, 'height', index)
#         x = self.dataset.growth_dataset(growth, spot, 'x', index)
#         y = self.dataset.growth_dataset(growth, spot, 'y', index)
#         width_x = self.dataset.growth_dataset(growth, spot, 'width_x', index)
#         width_y = self.dataset.growth_dataset(growth, spot, 'width_y', index)

#         print(f'img_sum:{img_sum:.2f}, img_max:{img_max:.2f}, img_mean:{img_mean:.2f}')
#         print(f'img_rec_sum:{img_rec_sum:.2f}, img_rec_max:{img_rec_max:.2f}, img_rec_mean:{img_rec_mean:.2f}')
#         print(f'height:{height:.2f}, x:{x:.2f}, y:{y:.2f}, width_x:{width_x:.2f}, width_y_max:{width_y:.2f}')

#         sample_list = [img, img_rec, img_rec-img]
#         fig, axes = layout_fig(3, 3, figsize=figsize)
#         for i, ax in enumerate(axes):
#             ax.imshow(sample_list[i])
#             labelfigs(ax, i)
#             ax.axis("off")

#         if filename is True: 
#             filename = f"RHEED_{self.dataset.sample_name}_{growth}_{spot}_{index}_img,img_rec,differerce"
                
#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename, **kwargs)
#         plt.show()
#         print('a: original, b: reconstructed image, c: difference')

#     def RHEED_parameter_trend(self, growth_list, spot, metric_list=None, filename = None, **kwargs):
    
#         if metric_list is None:
#             metric_list = ['img_sum', 'img_max', 'img_mean', 'img_rec_sum', 'img_rec_max', 'img_rec_mean', 'height', 'x', 'y', 'width_x', 'width_y']
#         for i, metric in enumerate(metric_list):
#             fig, ax = plt.subplots(figsize = (8, 1.5))
#             x_curve, y_curve = self.dataset.load_multiple_curves(growth_list, spot=spot, metric=metric, **kwargs)
#             # filter = abs(y_curve - np.mean(y_curve)) < 1 * np.std(y_curve)
#             # x_curve = x_curve[filter]
#             # y_curve = y_curve[filter]
            
#             ax.scatter(x_curve, y_curve, color='k', s=1)
#             set_labels(ax, xlabel='Time (s)', ylabel=f'{metric} (a.u.)')

#             # data_1d = parameters_all[:,i]
#             # data_1d = data_1d[abs(data_1d - np.mean(data_1d)) < 2 * np.std(data_1d)]
#             # plt.plot(data_1d, color='k')
#             # set_labels(ax, xlabel='Time (s)', ylabel='Intensity (a.u.)')
#             if filename: 
#                 filename = f"RHEED_{self.dataset.sample_name}_{spot}_{metric}"
                    
#             # prints the figure
#             if self.Printer is not None and filename is not None:
#                 self.Printer.savefig(fig, filename, **kwargs)
#             plt.show()

    
# def make_fine_step(x, transparency, step, color, saturation=1, savgol_filter_level=(15,1)):
#     x_FineStep = np.hstack([np.linspace(start, stop, num=step+1, endpoint=True)[:-1] for start, stop in zip(x, x[1:])])
    
#     transparency_FineStep = np.hstack([np.linspace(start, stop, num=step+1, endpoint=True)[:-1] for start, stop in zip(transparency, transparency[1:])])
#     if not isinstance(savgol_filter_level, type(None)):
#         transparency_FineStep_before = np.copy(transparency_FineStep)
#         transparency_FineStep = savgol_filter(transparency_FineStep, savgol_filter_level[0]*step+1, savgol_filter_level[1])

#     transparency_FineStep_norm = np.expand_dims((transparency_FineStep / max(transparency_FineStep)) * saturation, 1)
#     transparency_FineStep_norm[transparency_FineStep_norm<0] = 0
    
#     colors = np.repeat([[*color]], len(transparency_FineStep_norm), 0)
#     colors_all = np.concatenate([colors, transparency_FineStep_norm], 1)
#     return x_FineStep, colors_all


# def two_color_array(x_all, x1, x2, c1, c2, transparency=1):
#     '''
#     add docstring
#     '''
#     color_array = np.zeros([len(x_all), 4], dtype=np.float32)
#     color_array[np.isin(x_all, x1)] = [*c1, transparency]
#     color_array[np.isin(x_all, x2)] = [*c2, transparency]
#     return color_array

# def trim_axes(axs, N):
#     """
#     Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
#     """
#     axs = axs.flat
#     for ax in axs[N:]:
#         ax.remove()
#     return axs[:N]

# def show_images(images, labels=None, img_per_row=8, img_height=1, colorbar=False, 
#                 clim=False, scale_0_1=False, hist_bins=None, show_axis=False):
#     assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"


#     def scale(x):
#         if x.min() < 0:
#             return (x - x.min()) / (x.max() - x.min())
#         else:
#             return x/(x.max() - x.min())
    
#     h = images[0].shape[1] // images[0].shape[0]*img_height + 1
#     if not labels:
#         labels = range(len(images))
        
#     n = 1
#     if hist_bins: n +=1
        
#     fig, axes = plt.subplots(n*len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
#                              figsize=(16, n*h*len(images)//img_per_row+1))
#     trim_axes(axes, len(images))

#     for i, img in enumerate(images):
        
#         if scale_0_1: img = scale(img)
        
#         if len(images) <= img_per_row and not hist_bins:
#             index = i%img_per_row
#         else:
#             index = (i//img_per_row)*n, i%img_per_row

#         axes[index].title.set_text(labels[i])
#         im = axes[index].imshow(img)
#         if colorbar:
#             fig.colorbar(im, ax=axes[index])
            
#         if clim:
#             m, s = np.mean(img), np.std(img)            
#             im.set_clim(m-3*s, m+3*s) 
            
#         if not show_axis:
#             axes[index].axis('off')

#         if hist_bins:
#             index_hist = (i//img_per_row)*n+1, i%img_per_row
#             h = axes[index_hist].hist(img.flatten(), bins=hist_bins)
#     plt.show()


# def draw_background_colors(ax, bg_colors):
#     if isinstance(bg_colors, tuple):
#         ax.set_facecolor(bg_colors)
#     elif bg_colors is not None:
#         x_coor = bg_colors[:, 0]
#         colors = bg_colors[:, 1:]
#         for i in range(len(x_coor)):
#             if i == 0:
#                 end = (x_coor[i] + x_coor[i+1]) / 2
#                 start = end - (x_coor[i+1] - x_coor[i])
#             elif i == len(x_coor) - 1:
#                 start = (x_coor[i-1] + x_coor[i]) / 2
#                 end = start + (x_coor[i] - x_coor[i-1])
#             else:
#                 start = (x_coor[i-1] + x_coor[i]) / 2
#                 end = (x_coor[i] + x_coor[i+1]) / 2
#             ax.axvspan(start, end, facecolor=colors[i])
            
# def draw_boxes(ax, boxes, box_color):
#     for (box_start, box_end) in boxes:
#         ax.axvspan(box_start, box_end, facecolor=box_color, edgecolor=box_color)

# def find_nearest(array, value):
#     idx = np.abs(array - value).argmin()
#     return array[idx]

# # def plot_scatter(ax, curve_x, curve_y, color='k', markersize=1):
# #     ax.scatter(x=curve_x, y=curve_y, c=color, s=markersize)

# # def plot_lineplot(ax, curve_x, curve_y, color='k', markersize=1):
# #     ax.plot(curve_x, curve_y, color=color, marker='.', markersize=markersize)

# def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', 
#                logscale=False, legend=None):
#     if type(xlabel) != type(None): ax.set_xlabel(xlabel)
#     if type(ylabel) != type(None): ax.set_ylabel(ylabel)
#     if type(title) != type(None): ax.set_title(title)
#     if type(xlim) != type(None): ax.set_xlim(xlim)
#     if type(ylim) != type(None): ax.set_ylim(ylim)
#     if yaxis_style == 'sci':
#         plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
#     if logscale: plt.yscale("log") 
#     if legend: plt.legend(legend)

# def label_curves(ax, curve_x, curve_y, labels_dict):
#     if type(labels_dict) != type(None):
#         for x in labels_dict.keys():
#             y = curve_y[np.where(curve_x==find_nearest(curve_x, x))]
#             pl.text(x, y, str(labels_dict[x]), color="g", fontsize=6)
            
            
# # def plot_curve(curve_x, curve_y, curve_x_fit=None, curve_y_fit=None, plot_colors=['k', 'r'], plot_type='scatter', markersize=1, xlabel=None, ylabel=None, xlim=None, ylim=None, logscale=False, yaxis_style='sci', title=None, legend=None, figsize=(12,2.5), save_path=None):
    
# #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    
# #     if plot_type == 'scatter':
# #         plot_scatter(ax, curve_x, curve_y, plot_colors[0], markersize)
# #         if not isinstance(curve_y_fit, type(None)):
# #             if not isinstance(curve_x_fit, type(None)):
# #                 plot_scatter(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
# #             else:
# #                 plot_scatter(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                
# #     if plot_type == 'lineplot':
# #         plot_lineplot(ax, curve_x, curve_y, plot_colors[0], markersize)
# #         if not isinstance(curve_y_fit, type(None)):
# #             if not isinstance(curve_x_fit, type(None)):
# #                 plot_lineplot(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
# #             else:
# #                 plot_lineplot(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                
# #     set_labels(ax, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, yaxis_style=yaxis_style, 
# #                logscale=logscale, legend=legend)
# #     if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #     plt.show()

# def set_index(axes, index, total_length):
#     rows, img_per_row = axes.shape    
#     if total_length <= img_per_row:
#         index = index%img_per_row
#     else:
#         index = (index//img_per_row), index%img_per_row

# # fig, axes = plt.subplots(len(ys)//img_per_row+1*int(len(ys)%img_per_row>0), img_per_row, 
# #                          figsize=(16, subplot_height*len(ys)//img_per_row+1))  
# # def show_grid_plots(xs, ys, labels=None, ys_fit1=None, ys_fit2=None, img_per_row=4, subplot_height=3, ylim=None, legend=None):
# def show_grid_plots(axes, xs, ys, titles=None, xlabel=None, ylabel=None, ylim=None, legend=None):

#     if type(labels) == type(None): labels = range(len(ys))

# # in new version, axes should be trimmed already
#     # trim_axes(axes, len(ys))

# # get the img_per_row from axes:

#     for i in range(len(ys)):
#         i = set_index(axes, i)
        
#         im = axes[i].plot(xs[i], ys[i], marker='.', markersize=8, color=(44/255,123/255,182/255, 0.5))
#         # axes[index].title.set_text(labels[i])

#         # if type(ys_fit1) != type(None):
#         #     im = axes[index].plot(xs[i], ys_fit1[i], linewidth=4, color=(217/255,95/255,2/255))
            
#         # if type(ys_fit2) != type(None):
#         #     im = axes[index].plot(xs[i], ys_fit2[i], linewidth=2, color=(27/255,158/255,119/255))

#         set_labels(axes[i], xlabel=xlabel, ylabel=ylabel, title=titles[i], ylim=ylim, legend=legend)

#         # if type(ylim) != type(None):
#         #     axes[index].set_ylim([ylim[0], ylim[1]])
            
#         # if type(legend) != type(None): axes[index].legend(legend)

#     # fig.tight_layout()
#     plt.show()
    
    

# def visualize_predictions(x_all, y_all, all_outputs):
#     parameters_all, parameters_processed_all, x_coor_all, info = all_outputs
#     [xs_all, ys_all, xs_processed_all, ys_processed_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all] = info
    
#     x_coor_all = np.copy(x_coor_all)
#     parameters = np.copy(parameters_all)

#     # seperate RHEED data based on different function forms
#     ktop = parameters_processed_all[:,-2:]
#     x1_all, y1_all, x2_all, y2_all = [], [], [], []
#     for k, xx, yy in zip(ktop, xs_all, ys_all):
#         if k[0] == 0: 
#             x1_all.append(xx)
#             y1_all.append(yy)
#         if k[0] == 1: 
#             x2_all.append(xx)
#             y2_all.append(yy)
#     if x1_all != []: x1_all = np.concatenate(x1_all)
#     if y1_all != []: y1_all = np.concatenate(y1_all)
#     if x2_all != []: x2_all = np.concatenate(x2_all)
#     if y2_all != []: y2_all = np.concatenate(y2_all)

#     plot_curve(x_all, y_all, xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12,2.5), xlim=(-2, 135))

#     plot_curve(x1_all, y1_all, x2_all, y2_all, xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12,2.5), xlim=(-2, 135))

#     plot_curve(x_coor_all, parameters_processed_all[:,0], plot_type='lineplot', xlabel='Time (s)', ylabel='y1: a (a.u.)', 
#                yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
#     plot_curve(x_coor_all, parameters_processed_all[:,1], plot_type='lineplot', xlabel='Time (s)', ylabel='y1: b*x (a.u.)', 
#                yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
#     plot_curve(x_coor_all, parameters_processed_all[:,2], plot_type='lineplot', xlabel='Time (s)', ylabel='y1: c*x^2 (a.u.)', 
#                yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))

#     plot_curve(x_coor_all, parameters_processed_all[:,3], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: m1 (a.u.)', 
#                yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
#     plot_curve(x_coor_all, parameters_processed_all[:,4], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: m2*x (a.u.)', 
#                yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
#     plot_curve(x_coor_all, parameters_processed_all[:,5], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: Characteristic Time (s)', 
#                yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))

#     print('MSE loss for DL model fitting is:', np.mean((np.concatenate(ys_nor_all, 0)-np.concatenate(ys_nor_fit_all, 0))**2))

    
# def label_violinplot(ax, data, label_type='average', text_pos='center'):
#     '''
#     data: a list of list or numpy array for different 
#     '''
    
#     # Calculate number of obs per group & median to position labels
#     xloc = range(len(data))
#     yloc, text = [], [] 
    
#     for i, d in enumerate(data):
#         yloc.append(np.median(d))
        
#         if label_type == 'number':
#             text.append("n: "+str(len(d)))
        
#         if label_type == 'average':
#             text.append(str(round(np.median(d), 4)))

#     for tick, label in zip(xloc, ax.get_xticklabels()):
#         if text_pos == 'center':
#             ax.text(xloc[tick], yloc[tick]*1.1, text[tick], horizontalalignment='center', size=14, weight='semibold')
#         if text_pos == 'right':
#             ax.text(xloc[tick]+0.02, yloc[tick]*0.7, text[tick], horizontalalignment='left', size=14, weight='semibold')
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

sys.path.append('../../src')
from m3_learning.viz.layout import layout_fig, labelfigs
from m3_learning.RHEED.Viz import Viz
from m3_learning.RHEED.Analysis import analyze_curves, remove_outlier, smooth

def decay_curve_examples(df_para, spot, metric, fit_settings):
    """
    Plot decay curve examples.

    Args:
        df_para (DataFrame): Dataframe containing parameters.
        spot (str): Spot identifier.
        metric (str): Metric to analyze.
        fit_settings (dict): Settings for curve fitting.

    Returns:
        None
    """
    color_blue = (44/255,123/255,182/255)
    seq_colors = ['#00429d','#2e59a8','#4771b2','#5d8abd','#73a2c6','#8abccf','#a5d5d8','#c5eddf','#ffffe0']
    bgc1, bgc2 = (*colors.hex2color(seq_colors[0]), 0.3), (*colors.hex2color(seq_colors[5]), 0.3) 

    parameters_all, x_coor_all, info = analyze_curves(df_para, {'growth_1': 1}, spot, metric, interval=0, fit_settings=fit_settings)
    [xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all, labels_all, losses_all] = info
    sample_list = [6, 21]
    loc_list = ['ct', 'cb']
    fig, axes = layout_fig(2, 2, figsize=(5, 3))
    for i, ax in enumerate(axes):
        Viz.draw_background_colors(ax, ([bgc1, bgc2][i]))
        ax.scatter(xs_all[sample_list[i]], ys_nor_all[sample_list[i]], color=np.array(color_blue).reshape(1,-1), s=2)
        ax.scatter(xs_all[sample_list[i]], ys_nor_fit_all[sample_list[i]], color='k', s=2)
        ax.set_box_aspect(1)
        Viz.set_labels(ax, xlabel='Time (s)', ylabel='Intensity (a.u.)', yaxis_style='linear')
        labelfigs(ax, None, string_add=labels_all[sample_list[i]], loc=loc_list[i], style='b', size=6)
        


def compare_loss_difference():
    """
    Compare the difference in losses between two samples.

    Returns:
        None
    """

    # load data if needed
    x_all_sample1, y_all_sample1 = np.load('Saved_data/treated_213nm-x_all.npy'), np.load('Saved_data/treated_213nm-y_all.npy')
    color_array_sample1 = np.load('Saved_data/treated_213nm-bg_growth.npy')
    losses_all_sample1 = np.load('Saved_data/treated_213nm-losses_all.npy')
    loss_diff_sample1 = np.abs(losses_all_sample1[:,0] - losses_all_sample1[:,1])

    x_all_sample2, y_all_sample2 = np.load('Saved_data/treated_81nm-x_all.npy'), np.load('Saved_data/treated_81nm-y_all.npy')
    color_array_sample2 = np.load('Saved_data/treated_81nm-bg_growth.npy')
    losses_all_sample2 = np.load('Saved_data/treated_81nm-losses_all.npy')
    loss_diff_sample2 = np.abs(losses_all_sample2[:,0] - losses_all_sample2[:,1])

    x_all_sample3, y_all_sample3 = np.load('Saved_data/untreated_162nm-x_all.npy'), np.load('Saved_data/untreated_162nm-y_all.npy')
    color_array_sample3 = np.load('Saved_data/untreated_162nm-bg_growth.npy')
    losses_all_sample3 = np.load('Saved_data/untreated_162nm-losses_all.npy')
    loss_diff_sample3 = np.abs(losses_all_sample3[:,0] - losses_all_sample3[:,1])


    seq_colors = ['#00429d','#2e59a8','#4771b2','#5d8abd','#73a2c6','#8abccf','#a5d5d8','#c5eddf','#ffffe0']
    fig, axes = layout_fig(3, 1, figsize=(5, 1.5*3))
    Viz.plot_loss_difference(axes[0], x_all_sample1, y_all_sample1, color_array_sample1[:,0], loss_diff_sample1, 
                            color_array_sample1, color_2=seq_colors[0], title='treated_213nm')
    Viz.plot_loss_difference(axes[1], x_all_sample2, y_all_sample2, color_array_sample2[:,0], loss_diff_sample2, 
                            color_array_sample2, color_2=seq_colors[0], title='treated_81nm')
    Viz.plot_loss_difference(axes[2], x_all_sample3, y_all_sample3, color_array_sample3[:,0], loss_diff_sample3, 
                            color_array_sample3, color_2=seq_colors[0], title='untreated_162nm')
    plt.show()


def compare_growth_mechanism():
    """
    Compare the growth mechanism of different samples.

    Returns:
        None
    """

    # load data if needed
    x_all_sample1, y_all_sample1 = np.load('Saved_data/treated_213nm-x_all.npy'), np.load('Saved_data/treated_213nm-y_all.npy')
    color_array_sample1 = np.load('Saved_data/treated_213nm-bg_growth.npy')
    boxes_sample1 = np.load('Saved_data/treated_213nm-boxes.npy')

    x_all_sample2, y_all_sample2 = np.load('Saved_data/treated_81nm-x_all.npy'), np.load('Saved_data/treated_81nm-y_all.npy')
    color_array_sample2 = np.load('Saved_data/treated_81nm-bg_growth.npy')
    boxes_sample2 = np.load('Saved_data/treated_81nm-boxes.npy')

    x_all_sample3, y_all_sample3 = np.load('Saved_data/untreated_162nm-x_all.npy'), np.load('Saved_data/untreated_162nm-y_all.npy')
    color_array_sample3 = np.load('Saved_data/untreated_162nm-bg_growth.npy')
    boxes_sample3 = np.load('Saved_data/untreated_162nm-boxes.npy')
    
    color_gray = (128/255, 128/255, 128/255, 0.5)
    
    fig, axes = layout_fig(3, 1, figsize=(6, 2*3))
    Viz.draw_background_colors(axes[0], color_array_sample1)
    Viz.draw_boxes(axes[0], boxes_sample1, color_gray)
    axes[0].scatter(x_all_sample1, y_all_sample1, color='k', s=1)
    Viz.set_labels(axes[0], xlabel='Time (s)', ylabel='Intensity (a.u.)', title='treated_213nm')

    Viz.draw_background_colors(axes[1], color_array_sample2)
    Viz.draw_boxes(axes[1], boxes_sample2, color_gray)
    axes[1].scatter(x_all_sample2, y_all_sample2, color='k', s=1)
    Viz.set_labels(axes[1], xlabel='Time (s)', ylabel='Intensity (a.u.)', title='treated_81nm')

    Viz.draw_background_colors(axes[2], color_array_sample3)
    Viz.draw_boxes(axes[2], boxes_sample3, color_gray)
    axes[2].scatter(x_all_sample3, y_all_sample3, color='k', s=1)
    Viz.set_labels(axes[2], xlabel='Time (s)', ylabel='Intensity (a.u.)', title='untreated_162nm')
    plt.show()



def visualize_characteristic_time():
    """
    Visualize the characteristic time for different samples.

    Returns:
        None
    """
    seq_colors = ['#00429d','#2e59a8','#4771b2','#5d8abd','#73a2c6','#8abccf','#a5d5d8','#c5eddf','#ffffe0']
    fig, axes = layout_fig(3, 1, figsize=(6, 6))
    ax1, ax3, ax5 = axes[0], axes[1], axes[2]

    x_all_sample1, y_all_sample1 = np.load('Saved_data/treated_213nm-x_all.npy'), np.load('Saved_data/treated_213nm-y_all.npy')
    x_sklearn_sample1, tau_sklearn_sample1 = np.swapaxes(np.load('Saved_data/treated_213nm-fitting_results(sklearn).npy'), 0, 1)[[0, -1]]
    x_sklearn_sample1, tau_clean_sample1 = remove_outlier(x_sklearn_sample1, tau_sklearn_sample1, 0.95)
    tau_smooth_sample1 = smooth(tau_clean_sample1, 3)

    bg_growth_sample1 = np.load('Saved_data/treated_213nm-bg_growth.npy')
    Viz.draw_background_colors(ax1, bg_growth_sample1)
    ax1.scatter(x_all_sample1, y_all_sample1, color='k', s=1)
    Viz.set_labels(ax1, xlabel='Time (s)', ylabel='Intensity (a.u.)', xlim=(-2, 130), title='Treated_213nm', ticks_both_sides=False)

    ax2 = ax1.twinx()
    ax2.scatter(x_sklearn_sample1, tau_clean_sample1, color=seq_colors[0], s=3)
    ax2.plot(x_sklearn_sample1, tau_smooth_sample1, color='#bc5090', markersize=3)
    Viz.set_labels(ax2, ylabel='Characteristic Time (s)', yaxis_style='lineplot', ylim=(-0.05, 0.5), ticks_both_sides=False)
    ax2.tick_params(axis="y", color='k', labelcolor=seq_colors[0])
    ax2.set_ylabel('Characteristic Time (s)', color=seq_colors[0])
    ax2.legend(['original', 'processed'], fontsize=8)

    x_all_sample2, y_all_sample2 = np.load('Saved_data/treated_81nm-x_all.npy'), np.load('Saved_data/treated_81nm-y_all.npy')
    x_sklearn_sample2, tau_sklearn_sample2 = np.swapaxes(np.load('Saved_data/treated_81nm-fitting_results(sklearn).npy'), 0, 1)[[0, -1]]
    x_sklearn_sample2, tau_clean_sample2 = remove_outlier(x_sklearn_sample2, tau_sklearn_sample2, 0.95)
    tau_smooth_sample2 = smooth(tau_clean_sample2, 3)

    bg_growth_sample2 = np.load('Saved_data/treated_81nm-bg_growth.npy')
    
    Viz.draw_background_colors(ax3, bg_growth_sample2)
    ax3.scatter(x_all_sample2, y_all_sample2, color='k', s=1)
    Viz.set_labels(ax3, xlabel='Time (s)', ylabel='Intensity (a.u.)', xlim=(-2, 115), title='Treated_81nm', ticks_both_sides=False)

    ax4 = ax3.twinx()
    ax4.scatter(x_sklearn_sample2, tau_clean_sample2, color=seq_colors[0], s=3)
    ax4.plot(x_sklearn_sample2, tau_smooth_sample2, color='#bc5090', markersize=3)
    Viz.set_labels(ax4, ylabel='Characteristic Time (s)', yaxis_style='lineplot', ylim=(-0.05, 0.5), ticks_both_sides=False)
    ax4.tick_params(axis="y", color='k', labelcolor=seq_colors[0])
    ax4.set_ylabel('Characteristic Time (s)', color=seq_colors[0])
    ax4.legend(['original', 'processed'], fontsize=8)    

    x_all_sample3, y_all_sample3 = np.load('Saved_data/untreated_162nm-x_all.npy'), np.load('Saved_data/untreated_162nm-y_all.npy')
    x_sklearn_sample3, tau_sklearn_sample3 = np.swapaxes(np.load('Saved_data/untreated_162nm-fitting_results(sklearn).npy'), 0, 1)[[0, -1]]
    x_sklearn_sample3, tau_clean_sample3 = remove_outlier(x_sklearn_sample3, tau_sklearn_sample3, 0.95)
    tau_smooth_sample3 = smooth(tau_clean_sample3, 3)

    bg_growth_sample3 = np.load('Saved_data/untreated_162nm-bg_growth.npy')
    Viz.draw_background_colors(ax5, bg_growth_sample3)
    ax5.scatter(x_all_sample3, y_all_sample3, color='k', s=1)
    Viz.set_labels(ax5, xlabel='Time (s)', ylabel='Intensity (a.u.)', xlim=(-2, 125), title='Untreated_163nm', ticks_both_sides=False)

    ax6 = ax5.twinx()
    ax6.scatter(x_sklearn_sample3, tau_clean_sample3, color=seq_colors[0], s=3)
    ax6.plot(x_sklearn_sample3, tau_smooth_sample3, color='#bc5090', markersize=3)
    Viz.set_labels(ax6, ylabel='Characteristic Time (s)', yaxis_style='lineplot', ylim=(-0.05, 0.5), ticks_both_sides=False)
    ax6.tick_params(axis="y", color='k', labelcolor=seq_colors[0])
    ax6.set_ylabel('Characteristic Time (s)', color=seq_colors[0])
    ax6.legend(['original', 'processed'], fontsize=8) 
    plt.show()

def violinplot_characteristic_time():
    """
    Generate a violin plot of the characteristic time for different samples.

    Returns:
        None
    """
    color_blue = (44/255,123/255,182/255)
    color_orange = (217/255,95/255,2/255)
    color_purple = (117/255,112/255,179/255)

    x_all_sample1, y_all_sample1 = np.load('Saved_data/treated_213nm-x_all.npy'), np.load('Saved_data/treated_213nm-y_all.npy')
    x_sklearn_sample1, tau_sklearn_sample1 = np.swapaxes(np.load('Saved_data/treated_213nm-fitting_results(sklearn).npy'), 0, 1)[[0, -1]]
    x_sklearn_sample1, tau_clean_sample1 = remove_outlier(x_sklearn_sample1, tau_sklearn_sample1, 0.95)

    x_all_sample2, y_all_sample2 = np.load('Saved_data/treated_81nm-x_all.npy'), np.load('Saved_data/treated_81nm-y_all.npy')
    x_sklearn_sample2, tau_sklearn_sample2 = np.swapaxes(np.load('Saved_data/treated_81nm-fitting_results(sklearn).npy'), 0, 1)[[0, -1]]
    x_sklearn_sample2, tau_clean_sample2 = remove_outlier(x_sklearn_sample2, tau_sklearn_sample2, 0.95)

    x_all_sample3, y_all_sample3 = np.load('Saved_data/untreated_162nm-x_all.npy'), np.load('Saved_data/untreated_162nm-y_all.npy')
    x_sklearn_sample3, tau_sklearn_sample3 = np.swapaxes(np.load('Saved_data/untreated_162nm-fitting_results(sklearn).npy'), 0, 1)[[0, -1]]
    x_sklearn_sample3, tau_clean_sample3 = remove_outlier(x_sklearn_sample3, tau_sklearn_sample3, 0.95)

    fig, ax = plt.subplots(figsize=(6, 2), layout='compressed')
    titles = ['Treated substrate\n(step width=213±88nm)',
            'Treated substrate\n(step width=81±44nm)',
            'Untreated substrate\n(step width=162±83μm)']
    ax = sns.violinplot(data=[tau_clean_sample1, tau_clean_sample2, tau_clean_sample3], 
                        palette=[color_blue, color_orange, color_purple], linewidth=0.8)
    ax.set_xticklabels(titles)
    Viz.set_labels(ax, ylabel='Characteristic Time (s)', ticks_both_sides=False, yaxis_style='linear')
    Viz.label_violinplot(ax, [tau_clean_sample1, tau_clean_sample2, tau_clean_sample3], label_type='average', text_pos='right')
    plt.show()
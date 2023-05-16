import numpy as np
from m3_learning.viz.layout import layout_fig, inset_connector, add_box, subfigures, add_text_to_figure, get_axis_pos_inches, imagemap,  FigDimConverter, labelfigs, imagemap
from scipy.signal import resample
from scipy import fftpack
import matplotlib.pyplot as plt
from m3_learning.be.nn import SHO_Model
import m3_learning
from m3_learning.util.rand_util import get_tuple_names
import torch
import pandas as pd
import seaborn as sns

color_palette = {
    "LSQF_A": "#003f5c",
    "LSQF_P": "#444e86",
    "NN_A": "#955196",
    "NN_P": "#dd5182",
    "other": "#ff6e54",
    "other_2": "#ffa600"
}


class Viz:

    def __init__(self, dataset, Printer=None, verbose=False, labelfigs_=True):
        self.Printer = Printer
        self.dataset = dataset
        self.verbose = verbose
        self.labelfigs = labelfigs_

        self.SHO_labels = [{'title': "Amplitude",
                            'y_label': "Amplitude \n (Arb. U.)"
                            },
                           {'title': "Resonance Frequency",
                            'y_label': "Resonance Frequency \n (Hz)"
                            },
                           {'title': "Dampening",
                            'y_label': "Quality Factor \n (Arb. U.)"},
                           {'title': "Phase",
                            'y_label': "Phase \n (rad)"}
                           ]

        self.color_palette = color_palette

    def static_state_decorator(func):
        """Decorator that stops the function from changing the state

        Args:
            func (method): any method
        """
        def wrapper(*args, **kwargs):
            current_state = args[0].dataset.get_state
            out = func(*args, **kwargs)
            args[0].dataset.set_attributes(**current_state)
            return out
        return wrapper

    @static_state_decorator
    def raw_be(self,
               dataset,
               filename="Figure_1_random_cantilever_resonance_results"):
        """Plots the raw data and the BE waveform

        Args:
            dataset (_type_): BE dataset
            filename (str, optional): Name to save the file. Defaults to "Figure_1_random_cantilever_resonance_results".
        """

        # Select a random point and time step to plot
        pixel = np.random.randint(0, dataset.num_pix)
        voltagestep = np.random.randint(0, dataset.voltage_steps)

        # Plots the amplitude and phase for the selected pixel and time step
        fig, ax = layout_fig(5, 5, figsize=(5 * (5/3), 1.3))

        # constructs the BE waveform and plot
        be_voltagesteps = len(dataset.be_waveform) / \
            dataset.be_repeats

        # plots the BE waveform
        ax[0].plot(dataset.be_waveform[: int(be_voltagesteps)])
        ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")

        # plots the resonance graph
        resonance_graph = np.fft.fft(
            dataset.be_waveform[: int(be_voltagesteps)])
        fftfreq = fftpack.fftfreq(int(be_voltagesteps)) * \
            dataset.sampling_rate
        ax[1].plot(
            fftfreq[: int(be_voltagesteps) //
                    2], np.abs(resonance_graph[: int(be_voltagesteps) // 2])
        )
        ax[1].axvline(
            x=dataset.be_center_frequency,
            ymax=np.max(resonance_graph[: int(be_voltagesteps) // 2]),
            linestyle="--",
            color="r",
        )
        ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
        ax[1].set_xlim(
            dataset.be_center_frequency - dataset.be_bandwidth -
            dataset.be_bandwidth * 0.25,
            dataset.be_center_frequency + dataset.be_bandwidth +
            dataset.be_bandwidth * 0.25,
        )

        # manually set the x limits
        x_start = 120
        x_end = 140

        # plots the hysteresis waveform and zooms in
        ax[2].plot(dataset.hysteresis_waveform)

        ax_new = ax[2].inset_axes([0.5, 0.65, 0.48, 0.33])
        ax_new.plot(dataset.hysteresis_waveform)
        ax_new.set_xlim(x_start, x_end)
        ax_new.set_ylim(0, -15)

        inset_connector(fig, ax[2], ax_new,
                        [(x_start, 0), (x_end, 0)], [(x_start, 0), (x_end, 0)],
                        color='k', linestyle='--', linewidth=.5)

        add_box(ax[2], (x_start, 0, x_end, -15), edgecolor='k',
                linestyle='--', facecolor='none', linewidth=.5, zorder=10)

        ax[2].set_xlabel("Voltage Steps")
        ax[2].set_ylabel("Voltage (V)")

        # changes the state to get the magnitude spectrum
        dataset.scaled = False
        dataset.raw_format = "magnitude spectrum"
        dataset.measurement_state = 'all'
        dataset.resampled = False

        # gets the data for the selected pixel and time step
        data_ = dataset.raw_spectra(pixel, voltagestep)

        # plots the magnitude spectrum for and phase for the selected pixel and time step
        ax[3].plot(
            dataset.frequency_bin,
            data_[0].flatten(),
        )
        ax[3].set(xlabel="Frequency (Hz)",
                  ylabel="Amplitude (Arb. U.)", facecolor='none')
        ax2 = ax[3].twinx()
        ax2.plot(
            dataset.frequency_bin,
            data_[1].flatten(),
            "r",
        )
        ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        ax[3].set_zorder(ax2.get_zorder() + 1)

        dataset.raw_format = "complex"
        data_ = dataset.raw_spectra(pixel, voltagestep)

        # plots the real and imaginary components for the selected pixel and time step
        ax[4].plot(dataset.frequency_bin, data_[0].flatten(), label="Real")
        ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
        ax3 = ax[4].twinx()
        ax3.plot(
            dataset.frequency_bin, data_[1].flatten(), 'r', label="Imaginary")
        ax3.set(xlabel="Frequency (Hz)",
                ylabel="Imag (Arb. U.)", facecolor='none')
        ax[4].set_zorder(ax3.get_zorder() + 1)

        # prints the figure
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style='b')

    def SHO_hist(self, SHO_data, filename=None):
        """Plots the SHO hysterisis parameters

        Args:
            SHO_data (numpy): SHO fit results
            filename (str, optional): filename where to save the results. Defaults to "".
        """
        SHO_data = SHO_data.reshape(-1, 4)

        # check distributions of each parameter before and after scaling
        fig, axs = layout_fig(4, 4, figsize=(5.25, 1.25))

        for i, (ax, label) in enumerate(zip(axs.flat, self.SHO_labels)):
            ax.hist(SHO_data[:, i].flatten(), 100)
            if i == 0:
                ax.set(ylabel="counts")
            ax.set(xlabel=label['y_label'])
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            ax.set_box_aspect(1)

        if self.verbose:
            self.dataset.extraction_state

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=axs, style='b')

    def SHO_loops(self, data=None, filename="Figure_2_random_SHO_fit_results"):

        if data is None:
            pixel = np.random.randint(0, self.dataset.num_pix)
            data = self.dataset.SHO_fit_results(pixel)

        # plots the SHO fit results for the selected pixel
        fig, axs = layout_fig(4, 4, figsize=(5.5, 1.1))

        for i, (ax, label) in enumerate(zip(axs, self.SHO_labels)):
            ax.plot(self.dataset.dc_voltage, data[0, :, i])
            ax.set_ylabel(label['y_label'])

        if self.verbose:
            self.dataset.extraction_state

        # prints the figure
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=axs, style='b')

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.dataset, key, value)

        # this makes sure the setter is called
        if kwargs.get("noise"):
            self.noise = kwargs.get("noise")

    def get_freq_values(self, data):
        data = data.flatten()
        if len(data) == self.dataset.resampled_bins:
            x = resample(self.dataset.frequency_bin,
                         self.dataset.resampled_bins)
        elif len(data) == len(self.dataset.frequency_bin):
            x = self.dataset.frequency_bin
        else:
            raise ValueError(
                "original data must be the same length as the frequency bins or the resampled frequency bins")
        return x

    def get_voltagestep(self, voltage_step):
        if voltage_step is None:

            if self.dataset.measurement_state == 'on' or self.dataset.measurement_state == 'off':
                voltage_step = np.random.randint(
                    0, self.dataset.voltage_steps // 2)
            else:
                voltage_step = np.random.randint(0, self.dataset.voltage_steps)
        return voltage_step

    @static_state_decorator
    def fit_tester(self, true, predict, pixel=None, voltage_step=None, **kwargs):

        # if a pixel is not provided it will select a random pixel
        if pixel is None:

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltage_step = self.get_voltagestep(voltage_step)

        self.set_attributes(**predict)
        params = self.dataset.SHO_LSQF(pixel=pixel, voltage_step=voltage_step)

        print(true)

        self.raw_data_comparison(
            true, predict, pixel=pixel, voltage_step=voltage_step, fit_results=params, **kwargs)

    @static_state_decorator
    def nn_checker(self, state, filename=None,
                   pixel=None, voltage_step=None, legend=True, **kwargs):

        # if a pixel is not provided it will select a random pixel
        if pixel is None:

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltage_step = self.get_voltagestep(voltage_step)

        self.set_attributes(**state)

        data = self.dataset.raw_spectra(pixel=pixel, voltage_step=voltage_step)

        # plot real and imaginary components of resampled data
        fig = plt.figure(figsize=(3, 1.25), layout='compressed')
        axs = plt.subplot(111)

        self.dataset.raw_format = "complex"

        data, x = self.dataset.raw_spectra(
            pixel, voltage_step, frequency=True, **kwargs)

        axs.plot(x, data[0].flatten(), 'k',
                 label=self.dataset.label + " Real")
        axs.set_xlabel("Frequency (Hz)")
        axs.set_ylabel("Real (Arb. U.)")
        ax2 = axs.twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, data[1].flatten(), 'g',
                 label=self.dataset.label + " Imag")

        axes = [axs, ax2]

        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        if self.verbose:
            self.dataset.extraction_state

        if legend:
            fig.legend(bbox_to_anchor=(1., 1),
                       loc="upper right", borderaxespad=0.1)

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, style='b')

    @static_state_decorator
    def raw_data_comparison(self,
                            true,
                            predict=None,
                            filename=None,
                            pixel=None,
                            voltage_step=None,
                            legend=True,
                            **kwargs):

        self.set_attributes(**true)

        # plot real and imaginary components of resampled data
        fig, axs = layout_fig(2, 2, figsize=(5, 1.25))

        # if a pixel is not provided it will select a random pixel
        if pixel is None:

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltage_step = self.get_voltagestep(voltage_step)

        # sets the dataset state to grab the magnitude spectrum
        self.dataset.raw_format = "magnitude spectrum"

        data, x = self.dataset.raw_spectra(
            pixel, voltage_step, frequency=True)  # deleted **kwargs here might cause problems later

        axs[0].plot(x, data[0].flatten(), 'b',
                    label=self.dataset.label + " Amplitude")
        ax1 = axs[0].twinx()
        ax1.plot(x, data[1].flatten(), 'r',
                 label=self.dataset.label + " Phase")

        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.dataset.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs)
            axs[0].plot(x, data[0].flatten(), 'bo',
                        label=self.dataset.label + " Amplitude")
            ax1.plot(x, data[1].flatten(), 'ro',
                     label=self.dataset.label + " Phase")
            self.set_attributes(**true)

        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Amplitude (Arb. U.)")
        ax1.set_ylabel("Phase (rad)")

        self.dataset.raw_format = "complex"

        data, x = self.dataset.raw_spectra(
            pixel, voltage_step, frequency=True)
        # had to delete kwargs here too

        axs[1].plot(x, data[0].flatten(), 'k',
                    label=self.dataset.label + " Real")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Real (Arb. U.)")
        ax2 = axs[1].twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, data[1].flatten(), 'g',
                 label=self.dataset.label + " Imag")

        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.dataset.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs)
            axs[1].plot(x, data[0].flatten(), 'ko',
                        label=self.dataset.label + " Real")
            ax2.plot(x, data[1].flatten(), 'gs',
                     label=self.dataset.label + " Imag")
            self.set_attributes(**true)

        axes = [axs[0], axs[1], ax1, ax2]
        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        if self.verbose:
            print("True \n")
            self.set_attributes(**true)
            self.dataset.extraction_state
            if predict is not None:
                print("predicted \n")
                self.set_attributes(**predict)
                self.dataset.extraction_state

        if legend:
            fig.legend(bbox_to_anchor=(1., 1),
                       loc="upper right", borderaxespad=0.1)

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=[
                                 axs[0], axs[1]], style='b')

    @static_state_decorator
    def nn_validation(self, model,
                      data=None, unscaled=True,
                      pixel=None, voltage_step=None,
                      index=None,
                      legend=True,
                      filename=None,
                      **kwargs):

        # Makes the figure
        fig, axs = layout_fig(2, 2, figsize=(5, 1.25))

        # sets the dataset state to grab the magnitude spectrum
        state = {'raw_format': 'magnitude spectrum',
                 'resampled': True}
        self.set_attributes(**state)

        # if set to scaled it will change the label
        if unscaled:
            label = ''
        else:
            label = 'scaled'

        # if an index is not provided it will select a random index
        # it is also possible to use a voltage step
        if index is None:

            # if a voltage step is provided it will use the voltage step to grab a specific index
            if voltage_step is not None:

                # if a pixel is not provided it will select a random pixel
                if pixel is None:

                    # Select a random point and time step to plot
                    pixel = np.random.randint(0, self.dataset.num_pix)

                # gets the voltagestep with consideration of the current state
                voltage_step = self.get_voltagestep(voltage_step)

                # gets the data based on a specific pixel and voltagestep
                data, x = self.dataset.raw_spectra(
                    pixel, voltage_step, frequency=True, **kwargs)

                SHO_results = self.dataset.SHO_LSQF(pixel, voltage_step)

        # if a smaller manual dataset is provided it will use that
        if data is not None:

            # if an index is not provided it will select a random index
            if index is None:
                index = np.random.randint(0, data.shape[0])

            # grabs the data based on the index
            data = data[[index]]

            # gets the frequency values from the dataset
            x = self.get_freq_values(data[:, :, 0])

        # computes the prediction from the nn model
        pred_data, scaled_param, parm = model.predict(data)

        # unscales the data
        if unscaled:
            data_complex = self.dataset.raw_data_scaler.inverse_transform(data)
            pred_data_complex = self.dataset.raw_data_scaler.inverse_transform(
                pred_data.numpy())
        else:
            data_complex = data
            pred_data_complex = self.dataset.to_complex(pred_data.numpy())

        # computes the magnitude spectrum from the data
        data_magnitude = self.dataset.to_magnitude(data_complex)
        pred_data_magnitude = self.dataset.to_magnitude(pred_data_complex)

        # plots the data
        axs[0].plot(x, pred_data_magnitude[0].flatten(), 'b',
                    label=label + " Amplitude \n NN Prediction")
        ax1 = axs[0].twinx()
        ax1.plot(x, pred_data_magnitude[1].flatten(), 'r',
                 label=label + " Phase \n NN Prediction")

        axs[0].plot(x, data_magnitude[0].flatten(), 'bo',
                    label=label + " Amplitude")
        ax1.plot(x, data_magnitude[1].flatten(), 'ro',
                 label=label + " Phase")

        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Amplitude (Arb. U.)")
        ax1.set_ylabel("Phase (rad)")

        pred_data_complex = self.dataset.to_real_imag(pred_data_complex)
        data_complex = self.dataset.to_real_imag(data_complex)

        axs[1].plot(x, pred_data_complex[0].flatten(), 'k',
                    label=label + " Real \n NN Prediction")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Real (Arb. U.)")
        ax2 = axs[1].twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, pred_data_complex[1].flatten(), 'g',
                 label=label + " Imag \n NN Prediction")

        axs[1].plot(x, data_complex[0].flatten(), 'ko',
                    label=label + " Real")
        ax2.plot(x, data_complex[1].flatten(), 'gs',
                 label=label + " Imag")

        axes = [axs[0], axs[1], ax1, ax2]

        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        if legend:
            fig.legend(bbox_to_anchor=(1., 1),
                       loc="upper right", borderaxespad=0.1)

        if "SHO_results" in kwargs:
            if voltage_step is None:
                SHO_results = kwargs["SHO_results"][[index]]

                if unscaled:
                    SHO_results = self.dataset.SHO_scaler.inverse_transform(
                        SHO_results)

            SHO_results = SHO_results.squeeze()

            fig.text(
                0.5, -.05, f"LSQF: A = {SHO_results[0]:0.2e} ,  \u03C9 = {SHO_results[1]/1000:0.1f} Hz, Q = {SHO_results[2]:0.1f}, \u03C6 = {SHO_results[3]:0.2f} rad",
                ha='center', fontsize=6)

            parm = parm.detach().numpy().squeeze()

            fig.text(
                0.5, -.15, f"NN: A = {parm[0]:0.2e} ,  \u03C9 = {parm[1]/1000:0.1f} Hz, Q = {parm[2]:0.1f}, \u03C6 = {parm[3]:0.2f} rad",
                ha='center', fontsize=6)

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=[
                                 axs[0], axs[1]], style='b')

    @static_state_decorator
    def best_median_worst_reconstructions(self, model, true, SHO_values=None,
                                          labels=["NN", "LSQF"], unscaled=True,
                                          filename=None, **kwargs):
        gaps = (0.8, 0.9)
        size = (1.25, 1.25)

        fig, ax = subfigures(3, 3, gaps=gaps, size=size)

        dpi = fig.get_dpi()

        prediction, scaled_param, parm = model.predict(true)

        index, mse = model.mse_rankings(true, prediction)

        ind = np.hstack(
            (index[0:3], index[len(index)//2-1:len(index)//2+2], index[-3:]))
        mse = np.hstack(
            (mse[0:3], mse[len(index)//2-1:len(index)//2+2], mse[-3:]))

        x = self.get_freq_values(true[0, :, 0])

        # unscales the data
        if unscaled:
            true = self.dataset.raw_data_scaler.inverse_transform(true)
            prediction = self.dataset.raw_data_scaler.inverse_transform(
                prediction.numpy())

            if self.dataset.raw_format == "magnitude spectrum":
                # computes the magnitude spectrum from the data
                true = self.dataset.to_magnitude(true)
                prediction = self.dataset.to_magnitude(prediction)
            else:
                # computes the magnitude spectrum from the data
                true = self.dataset.to_real_imag(true)
                prediction = self.dataset.to_real_imag(prediction)

            SHO_values = self.dataset.SHO_scaler.inverse_transform(SHO_values)

        else:
            true = true
            prediction = self.dataset.to_complex(prediction.numpy())

        ax.reverse()

        for i, (ind_, ax_) in enumerate(zip(ind, ax)):

            ax_.plot(x, prediction[0][ind_].flatten(), 'b',
                     label=labels[0] + " Amplitude")
            ax1 = ax_.twinx()
            ax1.plot(x, prediction[1][ind_].flatten(), 'r',
                     label=labels[0] + " Phase")

            ax_.plot(x, true[0][ind_].flatten(), 'bo',
                     label="Raw Amplitude")
            ax1.plot(x, true[1][ind_].flatten(), 'ro',
                     label="Raw Phase")

            ax_.set_xlabel("Frequency (Hz)")
            ax_.set_ylabel("Amplitude (Arb. U.)")
            ax1.set_ylabel("Phase (rad)")

            # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
            text_position_in_inches = (
                (gaps[0] + size[0])*(i % 3), (gaps[1] + size[1])*(3-i//3 - 1)-0.8)
            text = f'MSE: {mse[i]:0.4f}\nA LSQF:{SHO_values[ind_, 0]:0.2e} NN:{parm[ind_, 0]:0.2e}\n\u03C9: LSQF: {SHO_values[ind_, 1]/1000:0.1f} NN: {parm[ind_, 1]/1000:0.1f} Hz\nQ: LSQF: {SHO_values[ind_, 2]:0.1f} NN: {parm[ind_, 2]:0.1f}\n\u03C6: LSQF: {SHO_values[ind_, 3]:0.2f} NN: {parm[ind_, 3]:0.1f} rad'
            add_text_to_figure(fig, text, text_position_in_inches, fontsize=6)

            if i == 2:
                lines, labels = ax_.get_legend_handles_labels()
                lines2, labels2 = ax1.get_legend_handles_labels()
                ax_.legend(lines + lines2, labels + labels2, loc='upper right')

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, style='b')

    @static_state_decorator
    def get_best_median_worst(self,
                              true_state,
                              prediction=None,
                              out_state=None,
                              n=1,
                              SHO_results=False,
                              index=None,
                              **kwargs):

        if type(true_state) is dict:

            self.set_attributes(**true_state)

            # the data must be scaled to rank the results
            self.dataset.scaled = True

            true, x1 = self.dataset.raw_spectra(frequency=True)

        # condition if x_data is passed
        else:

            # converts to a standard form which is a list
            true = self.dataset.to_real_imag(true_state)

            try:
                # converts to numpy from tensor
                true = [data.numpy() for data in true]
            except:
                pass

            # gets the frequency values
            if true[0].ndim == 2:
                x1 = self.dataset.get_freq_values(true[0].shape[1])

        # holds the raw state
        current_state = self.dataset.get_state

        if isinstance(prediction, m3_learning.be.nn.SHO_Model):

            fitter = "NN"

            # sets the phase shift to zero for parameters
            # This is important if doing the fits because the fits will be wrong if the phase is shifted.
            self.dataset.NN_phase_shift = 0

            data = self.dataset.to_nn(true)

            pred_data, scaled_params, params = prediction.predict(data)

            self.dataset.scaled = True

            prediction, x2 = self.dataset.raw_spectra(
                fit_results=params, frequency=True)

        elif isinstance(prediction, dict):

            fitter = prediction["fitter"]

            exec(f"self.dataset.{prediction['fitter']}_phase_shift =0")

            self.dataset.scaled = False

            params = self.dataset.SHO_fit_results()

            self.dataset.scaled = True

            prediction, x2 = self.dataset.raw_spectra(
                fit_results=params, frequency=True)

        if "x2" not in locals():

            # if you do not use the model will run the
            x2 = self.dataset.get_freq_values(prediction[0].shape[1])

        # index the data if provided
        if index is not None:
            true = [true[0][index], true[1][index]]
            prediction = [prediction[0][index], prediction[1][index]]
            # params = params[index]

        # this must take the scaled data
        index1, mse1, d1, d2 = SHO_Model.get_rankings(true, prediction, n=n)

        d1, labels = self.out_state(d1, out_state)
        d2, labels = self.out_state(d2, out_state)

        # saves just the parameters that are needed
        params = params[index1]

        # resets the current state to apply the phase shifts
        self.set_attributes(**current_state)

        # gets the original index values
        if index is not None:
            index1 = index[index1]

        # if statement that will return the values for the SHO Results
        if SHO_results:
            if eval(f"self.dataset.{fitter}_phase_shift") is not None:
                params[:, 3] = eval(
                    f"self.dataset.shift_phase(params[:, 3], self.dataset.{fitter}_phase_shift)")
            return (d1, d2, x1, x2, labels, index1, mse1, params)
        else:
            return (d1, d2, x1, x2, labels, index1, mse1)

    def out_state(self, data, out_state):
        # holds the raw state
        current_state = self.dataset.get_state

        def convert_to_mag(data):
            data = self.dataset.to_complex(data, axis=1)
            data = self.dataset.raw_data_scaler.inverse_transform(data)
            data = self.dataset.to_magnitude(data)
            data = np.array(data)
            data = np.rollaxis(data, 0, data.ndim-1)
            return data

        labels = ["real", "imaginary"]

        if out_state is not None:

            if "raw_format" in out_state.keys():
                if out_state["raw_format"] == "magnitude spectrum":
                    data = convert_to_mag(data)
                    labels = ["Amplitude", "Phase"]

            elif "scaled" in out_state.keys():
                if out_state["scaled"] == False:
                    data = self.dataset.raw_data_scaler.inverse_transform(data)
                    labels = ["Scaled " + s for s in labels]

        self.set_attributes(**current_state)

        return data, labels

    @static_state_decorator
    def get_mse_index(self, index, model):

        # gets the raw data
        # returns the raw spectra in (samples, voltage steps, real/imaginary)
        data, _ = self.dataset.NN_data()

        # gets the index of the data selected
        # (samples, voltage steps, real/imaginary)
        data = data[[index]]

        if isinstance(model, m3_learning.be.nn.SHO_Model):

            # gets the predictions from the neural network
            predictions, params_scaled, params = model.predict(data)

            # detaches the tensor and converts to numpy
            predictions = predictions.detach().numpy()

        if isinstance(model, dict):

            # holds the raw state
            current_state = self.dataset.get_state

            # sets the phase shift to zero for the specific fitter - this is a requirement for using the fitting function
            exec(f"self.dataset.{model['fitter']}_phase_shift =0")

            # Ensures that we get the unscaled parameters
            # Only the unscaled parameters can be used to calculate the raw data
            self.dataset.scaled = False

            # Gets the parameters
            params = self.dataset.SHO_fit_results()

            # sets the dataset to scaled
            # we compare the MSE using the scaled parameters
            self.dataset.scaled = True

            # Ensures that the measurement state is complex
            self.dataset.raw_format = "complex"

            # This returns the raw data based on the parameters
            # this returns a list of the real and imaginary data
            pred_data = self.dataset.raw_spectra(
                fit_results=params)

            # makes the data an array
            # (real/imaginary, samples, voltage steps)
            pred_data = np.array(pred_data)

            # rolls the axis to (samples, voltage steps, real/imaginary)
            pred_data = np.rollaxis(pred_data, 0, pred_data.ndim)

            # gets the index of the data selected
            # (samples, voltage steps, real/imaginary)
            predictions = pred_data[[index]]

            # restores the state to the original state
            self.set_attributes(**current_state)

        return SHO_Model.MSE(data.detach().numpy(), predictions)

    @static_state_decorator
    def SHO_Fit_comparison(self,
                           data,
                           names,
                           gaps=(.8, .9),
                           size=(1.25, 1.25), model_comparison=None,
                           out_state=None,
                           filename=None,
                           display_results="all",
                           **kwargs):

        # gets the number of fits
        num_fits = len(data)

        # changes the gaps if the results are displayed
        if display_results == "MSE":
            gaps = (.8, .45)
        elif display_results is None:
            gaps = (.8, .33)

        # builds the subfigures
        fig, ax = subfigures(3, num_fits, gaps=gaps, size=size)

        # loops around the number of fits, and the data
        for step, (data, name) in enumerate(zip(data, names)):

            # unpack the data
            d1, d2, x1, x2, label, index1, mse1, params = data

            # loops around the datasets to compare
            for bmw, (true, prediction, error, SHO, index1) in enumerate(zip(d1, d2, mse1, params, index1)):

                # builds an empty dictionary to hold the errors, SHOs
                errors = {}
                SHOs = {}

                # selects the graph where the data is plot (rows, columns)
                i = bmw * num_fits + step

                ax_ = ax[i]
                ax_.plot(x2, prediction[0].flatten(), color=color_palette[f"{name}_A"],
                         label=f"{name} {label[0]}")
                ax1 = ax_.twinx()
                ax1.plot(x2, prediction[1].flatten(), color=color_palette[f"{name}_P"],
                         label=f"{name} {label[1]}")

                ax_.plot(x1, true[0].flatten(), 'o', color=color_palette["LSQF_A"],
                         label=f"Raw {label[0]}")
                ax1.plot(x1, true[1].flatten(), 'o', color=color_palette["LSQF_P"],
                         label=f"Raw {label[1]}")

                # saves error to the correct error name
                errors[name] = error
                SHOs[name] = SHO

                if model_comparison is not None:
                    if model_comparison[step] is not None:

                        pred_data, params, labels = self.get_SHO_params(index1,
                                                                        model=model_comparison[step],
                                                                        out_state=out_state)

                        # checks if using a neural network and saves the error
                        if isinstance(model_comparison[step], m3_learning.be.nn.SHO_Model):

                            # saves the color prefix
                            color = "NN"

                        # if the model is a dictionary then it is an LSQF model
                        if isinstance(model_comparison[step], dict):

                            # saves the color prefix
                            color = "LSQF"

                        # saves error to the correct error name
                        errors[color] = self.get_mse_index(
                            index1, model_comparison[step])
                        # might need to turn this into a numpy array and squeeze it
                        SHOs[color] = np.array(params).squeeze()

                        # plots the comparison graph
                        ax_.plot(x2, pred_data.squeeze()[0].flatten(), color=color_palette[f"{color}_A"],
                                 label=f"{color} {labels[0]}")
                        ax1.plot(x2, pred_data.squeeze()[1].flatten(), color=color_palette[f"{color}_P"],
                                 label=f"{color} {labels[1]}")
                        if display_results == "all":
                            error_string = f"MSE - LSQF: {errors['LSQF']:0.4f} NN: {errors['NN']:0.4f}\n AMP - LSQF:{SHOs['LSQF'][0]:0.2e} NN:{SHOs['NN'][0]:0.2e}\n\u03C9 - LSQF: {SHOs['LSQF'][1]/1000:0.1f} NN: {SHOs['NN'][1]/1000:0.1f} Hz\nQ- LSQF: {SHOs['LSQF'][2]:0.1f} NN: {SHOs['NN'][2]:0.1f}\n\u03C6- LSQF: {SHOs['LSQF'][3]:0.2f} NN: {SHOs['NN'][3]:0.1f} rad"
                        elif display_results == "MSE":
                            error_string = f"MSE - LSQF: {errors['LSQF']:0.4f} NN: {errors['NN']:0.4f}"

                # sets the xlabel, this is always frequency (HZ)
                ax_.set_xlabel("Frequency (Hz)")

                # if wants to display the results
                if display_results is not None:
                    # gets the axis position in inches - gets the bottom center
                    center = get_axis_pos_inches(fig, ax[i])

                    # selects the text position as an offset from the bottom center
                    text_position_in_inches = (center[0], center[1] - 0.33)

                    if "error_string" not in locals():
                        error_string = f'MSE: {error:0.4f}'

                    add_text_to_figure(
                        fig, error_string,
                        text_position_in_inches,
                        fontsize=6, ha='center', va='top',)

                if out_state is not None:
                    if "raw_format" in out_state.keys():
                        if out_state["raw_format"] == "magnitude spectrum":
                            ax_.set_ylabel("Amplitude (Arb. U.)")
                            ax1.set_ylabel("Phase (rad)")
                    else:
                        ax_.set_ylabel("Real (Arb. U.)")
                        ax1.set_ylabel("Imag (Arb. U.)")

                if i < num_fits:
                    # add a legend just for the last one
                    lines, labels = ax_.get_legend_handles_labels()
                    lines2, labels2 = ax1.get_legend_handles_labels()
                    ax_.legend(lines + lines2, labels +
                               labels2, loc='upper right')

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style='b')

    @static_state_decorator
    def get_SHO_params(self, index, model, out_state):
        """Function that gets the SHO parameters for a given index based on a specific model

        Args:
            index (list): list of indexes to get the SHO parameters for
            model (any): model or description of model that is used to compute the SHO results. 
            out_state (dict): dictionary that specifies the output state of the data.

        Returns:
            array, array, list: returns the output data, the SHO parameters, and the labels for the data
        """

        current_state = self.dataset.get_state

        pixel, voltage = np.unravel_index(
            index, (self.dataset.num_pix, self.dataset.voltage_steps))

        if isinstance(model, m3_learning.be.nn.SHO_Model):

            X_data, Y_data = self.dataset.NN_data()

            X_data = X_data[[index]]

            pred_data, scaled_param, params = model.predict(X_data)

            pred_data = np.array(pred_data)

        if isinstance(model, dict):

            # holds the raw state
            current_state = self.dataset.get_state

            self.dataset.scaled = False

            params_shifted = self.dataset.SHO_fit_results()

            exec(f"self.dataset.{model['fitter']}_phase_shift =0")

            params = self.dataset.SHO_fit_results()

            self.dataset.scaled = True

            pred_data = self.dataset.raw_spectra(
                fit_results=params)

            # output (channels, samples, voltage steps)
            pred_data = np.array([pred_data[0], pred_data[1]])

            # output (samples, channels, voltage steps)
            pred_data = np.swapaxes(pred_data, 0, 1)

            # output (samples, voltage steps, channels)
            pred_data = np.swapaxes(pred_data, 1, 2)

            params_shifted = params_shifted.reshape(-1, 4)

            pred_data = pred_data[[index]]
            params = params_shifted[[index]]

            self.set_attributes(**current_state)

        pred_data = np.swapaxes(pred_data, 1, 2)

        pred_data, labels = self.out_state(pred_data, out_state)

        # returns the state to the original state
        self.set_attributes(**current_state)

        return pred_data, params, labels

    @static_state_decorator
    def bmw_nn(self, true_state,
               prediction=None,
               model=None,
               out_state=None,
               n=1,
               gaps=(.8, .33),
               size=(1.25, 1.25),
               filename=None,
               **kwargs):

        d1, d2, x1, x2, label, index1, mse1 = self.get_best_median_worst(
            true_state,
            prediction=prediction,
            model=model,
            out_state=out_state,
            n=n,
            **kwargs)

        fig, ax = subfigures(1, 3, gaps=gaps, size=size)

        for i, (true, prediction, error) in enumerate(zip(d1, d2, mse1)):

            ax_ = ax[i]
            ax_.plot(x2, prediction[0].flatten(), color_palette["NN_A"],
                     label=f"NN {label[0]}")
            ax1 = ax_.twinx()
            ax1.plot(x2, prediction[1].flatten(), color_palette["NN_P"],
                     label=f"NN {label[1]}]")

            ax_.plot(x1, true[0].flatten(), 'o', color=color_palette["NN_A"],
                     label=f"Raw {label[0]}")
            ax1.plot(x1, true[1].flatten(), 'o', color=color_palette["NN_P"],
                     label=f"Raw {label[1]}")

            ax_.set_xlabel("Frequency (Hz)")

            # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
            text_position_in_inches = (
                -1*(gaps[0] + size[0])*((2-i) % 3) + size[0]/2, (gaps[1] + size[1])*(1.25-i//3-1.25) - gaps[1])
            text = f'MSE: {error:0.4f}'
            add_text_to_figure(
                fig, text, text_position_in_inches, fontsize=6, ha='center')

            if out_state is not None:
                if "measurement state" in out_state.keys():
                    if out_state["raw_format"] == "magnitude spectrum":
                        ax_.set_ylabel("Amplitude (Arb. U.)")
                        ax1.set_ylabel("Phase (rad)")
                else:
                    ax_.set_ylabel("Real (Arb. U.)")
                    ax1.set_ylabel("Imag (Arb. U.)")

        # add a legend just for the last one
        lines, labels = ax_.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax_.legend(lines + lines2, labels + labels2, loc='upper right')

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style='b')

        if "returns" in kwargs.keys():
            if kwargs["returns"] == True:
                return d1, d2, index1, mse1

    @static_state_decorator
    def SHO_switching_maps(self,
                           SHO_,
                           colorbars=True,
                           clims=[(0, 1.4e-4),  # amplitude
                                  (1.31e6, 1.33e6),  # resonance frequency
                                  (-230, -160),  # quality factor
                                  (-np.pi, np.pi)],  # phase
                           measurement_state="off",  # sets the measurement state to get the data
                           cycle=2,  # sets the cycle to get the data
                           cols=3,
                           fig_width=6.5,  # figure width in inches
                           number_of_steps=9,  # number of steps on the graph
                           voltage_plot_height=1.25,  # height of the voltage plot
                           intra_gap=0.02,  # gap between the graphs,
                           inter_gap=0.05,  # gap between the graphs,
                           cbar_gap=.5,  # gap between the graphs of colorbars
                           cbar_space=1.3,  # space on the right where the cbar is not
                           filename=None,
                           ):

        # sets the voltage state to off, and the cycle to get
        self.dataset.measurement_state = measurement_state
        self.dataset.cycle = cycle

        # instantiates the list of axes
        ax = []

        # number of rows
        rows = np.ceil(number_of_steps / 3)

        # calculates the size of the embedding image
        embedding_image_size = (fig_width - (inter_gap * (cols - 1)) -
                                intra_gap * 3 * cols - cbar_space*colorbars) / (cols * 4)

        # calculates the figure height based on the image details
        fig_height = rows * (embedding_image_size +
                             inter_gap) + voltage_plot_height + .33

        # defines a scalar to convert inches to relative coordinates
        fig_scalar = FigDimConverter((fig_width, fig_height))

        # creates the figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # left bottom width height
        pos_inch = [0.33, fig_height - voltage_plot_height,
                    6.5 - .33, voltage_plot_height]

        # adds the plot for the voltage
        ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

        # resets the x0 position for the embedding plots
        pos_inch[0] = 0
        pos_inch[1] -= embedding_image_size + 0.33

        # sets the embedding size of the image
        pos_inch[2] = embedding_image_size
        pos_inch[3] = embedding_image_size

        # adds the embedding plots
        for i in range(number_of_steps):

            # loops around the amp, phase, and freq
            for j in range(4):

                # adds the plot to the figure
                ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

                # adds the inter plot gap
                pos_inch[0] += embedding_image_size + intra_gap

            # if the last column in row, moves the position to the next row
            if (i+1) % cols == 0 and i != 0:

                # resets the x0 position for the embedding plots
                pos_inch[0] = 0

                # moves the y0 position to the next row
                pos_inch[1] -= embedding_image_size + inter_gap
            else:
                # adds the small gap between the plots
                pos_inch[0] += inter_gap

        # gets the DC voltage data - this is for only the on state or else it would all be 0
        voltage = self.dataset.dc_voltage

        # gets just part of the loop
        if hasattr(self.dataset, 'cycle') and self.dataset.cycle is not None:
            # gets the cycle of interest
            voltage = self.dataset.get_cycle(voltage)

        # gets the index of the voltage steps to plot
        inds = np.linspace(0, len(voltage)-1, number_of_steps, dtype=int)

        # converts the data to a numpy array
        if isinstance(SHO_, torch.Tensor):
            SHO_ = SHO_.detach().numpy()

        SHO_ = SHO_.reshape(self.dataset.num_pix,
                            self.dataset.voltage_steps, 4)

        # get the selected measurement cycle
        SHO_ = self.dataset.get_measurement_cycle(SHO_, axis=1)

        # plots the voltage
        ax[0].plot(voltage, "k")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].set_xlabel("Step")

        # Plot the data with different markers
        for i, ind in enumerate(inds):
            # this adds the labels to the graphs
            ax[0].plot(ind, voltage[ind], 'o', color='k', markersize=10)
            vshift = (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]) * .25

            # positions the location of the labels
            if voltage[ind] - vshift - .15 < ax[0].get_ylim()[0]:
                vshift = -vshift/2

            # adds the text to the graphs
            ax[0].text(ind, voltage[ind] - vshift,
                       str(i+1), color="k", fontsize=12)

        names = ["A", "\u03C9", "Q", "\u03C6"]

        for i, ind in enumerate(inds):

            # loops around the amp, resonant frequency, and Q, Phase
            for j in range(4):
                imagemap(ax[i*4+j+1], SHO_[:, ind, j],
                         colorbars=False, cmap="viridis",)

                if i // rows == 0:
                    labelfigs(ax[i*4+j+1], string_add=names[j],
                              loc="cb", size=5, inset_fraction=.2)

                ax[i*4+j+1].images[0].set_clim(clims[j])
            labelfigs(ax[1::4][i], string_add=str(i+1),
                      size=5, loc="bl", inset_fraction=.2)

        # if add colorbars
        if colorbars:

            # builds a list to store the colorbar axis objects
            bar_ax = []

            # gets the voltage axis position in ([xmin, ymin, xmax, ymax]])
            voltage_ax_pos = fig_scalar.to_inches(
                np.array(ax[0].get_position()).flatten())

            # loops around the 4 axis
            for i in range(4):

                # calculates the height and width of the colorbars
                cbar_h = (voltage_ax_pos[1] -
                          inter_gap - 2 * intra_gap - .33)/2
                cbar_w = (cbar_space - inter_gap - 2 * cbar_gap)/2

                # sets the position of the axis in inches
                pos_inch = [voltage_ax_pos[2] - (2 - i % 2)*(cbar_gap + cbar_w) + inter_gap,
                            voltage_ax_pos[1] - (i//2) *
                            (inter_gap + cbar_h) - .33 - cbar_h,
                            cbar_w,
                            cbar_h]

                # adds the plot to the figure
                bar_ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

                # adds the colorbars to the plots
                cbar = plt.colorbar(ax[i+1].images[0],
                                    cax=bar_ax[i], format='%.1e')
                cbar.set_label(names[i])  # Add a label to the colorbar

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(
                fig, filename, label_figs=ax[1::4], size=6, loc='tl', inset_fraction=.2)

        fig.show()

    @static_state_decorator
    def noisy_datasets(self, state, noise_level=None, pixel=None,
                       voltage_step=None, filename=None):

        if pixel is None:
            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        if voltage_step is None:
            voltage_step = np.random.randint(0, self.dataset.voltage_steps)

        self.set_attributes(**state)

        if noise_level is None:
            datasets = np.arange(0, len(self.dataset.raw_datasets))
        else:
            datasets = noise_level

        fig, ax_ = layout_fig(len(datasets), 4,
                              figsize=(4*(1.25 + .33), ((1+len(datasets))//4)*1.25))

        for i, (ax, noise) in enumerate(zip(ax_, datasets)):

            self.dataset.noise = noise

            data, x = self.dataset.raw_spectra(
                pixel, voltage_step, frequency=True)

            ax.plot(x, data[0].flatten(), color='k')
            ax1 = ax.twinx()
            ax1.plot(x, data[1].flatten(), color='b')

            ax.set_xlabel("Frequency (Hz)")

            if self.dataset.raw_format == "magnitude spectrum":
                ax.set_ylabel("Amplitude (Arb. U.)")
                ax1.set_ylabel("Phase (rad)")
            elif self.dataset.raw_format == "complex":
                ax.set_ylabel("Real (Arb. U.)")
                ax1.set_ylabel("Imag (Arb. U.)")

            # makes the box square
            ax.set_box_aspect(1)

            labelfigs(
                ax1, string_add=f"Noise {noise}", loc="ct", size=5, inset_fraction=.2, style='b')

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=ax_,
                                 size=6, loc='bl', inset_fraction=.2)

    @static_state_decorator
    def violin_plot_comparison(self, state, model, X_data, filename):

        self.set_attributes(**state)

        df = pd.DataFrame()

        # uses the model to get the predictions
        pred_data, scaled_param, params = model.predict(X_data)

        # scales the parameters
        scaled_param = self.dataset.SHO_scaler.transform(params)

        # gets the parameters from the SHO LSQF fit
        true = self.dataset.SHO_fit_results().reshape(-1, 4)

        # Builds the dataframe for the violin plot
        true_df = pd.DataFrame(
            true, columns=["Amplitude", "Resonance", "Q-Factor", "Phase"])
        predicted_df = pd.DataFrame(
            scaled_param, columns=["Amplitude",
                                   "Resonance", "Q-Factor", "Phase"]
        )

        # merges the two dataframes
        df = pd.concat((true_df, predicted_df))

        # adds the labels to the dataframe
        names = [true, scaled_param]
        names_str = ["LSQF", "NN"]
        # ["Amplitude", "Resonance", "Q-Factor", "Phase"]
        labels = ["A", "\u03C9", "Q", "\u03C6"]

        # adds the labels to the dataframe
        for j, name in enumerate(names):
            for i, label in enumerate(labels):
                dict_ = {
                    "value": name[:, i],
                    "parameter": np.repeat(label, name.shape[0]),
                    "dataset": np.repeat(names_str[j], name.shape[0]),
                }

                df = pd.concat((df, pd.DataFrame(dict_)))

        # builds the plot
        fig, ax = plt.subplots(figsize=(2, 2))

        # plots the data
        sns.violinplot(data=df, x="parameter", y="value",
                       hue="dataset", split=True, ax=ax)

        # labels the figure and does some styling
        labelfigs(ax, 0, style='b')
        ax.set_ylabel('Scaled SHO Results')
        ax.set_xlabel('')

        # Get the legend associated with the plot
        legend = ax.get_legend()
        legend.set_title("")

        # ax.set_aspect(1)

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename)

    def SHO_fit_movie_images(self, noise=None, comparison=None, fig_width=6.5,
                             voltage_plot_height=1.25,  # height of the voltage plot
                             intra_gap=0.02,  # gap between the graphs,
                             inter_gap=0.2,  # gap between the graphs,
                             cbar_gap=.5,  # gap between the graphs of colorbars
                             # space on the right where the cbar is not):
                             cbar_space=1.3,
                             colorbars=True,
                             ):

        # instantiates the list of axes
        ax = []

        # number of rows
        rows = 2
        cols = 4

        if comparison is not None:
            rows *= 2

        # calculates the size of the embedding image
        embedding_image_size = (fig_width - inter_gap -
                                intra_gap * 2 - cbar_space*colorbars) / (4)

        # calculates the figure height based on the image details
        fig_height = rows * (embedding_image_size +
                             inter_gap/2 + intra_gap/2) + voltage_plot_height + .33

        # defines a scalar to convert inches to relative coordinates
        fig_scalar = FigDimConverter((fig_width, fig_height))

        # creates the figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # left bottom width height
        pos_inch = [0.33, fig_height - voltage_plot_height,
                    6.5 - .33, voltage_plot_height]

        # adds the plot for the voltage
        ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

        # resets the x0 position for the embedding plots
        pos_inch[0] = 0
        pos_inch[1] -= embedding_image_size + 0.33

        # sets the embedding size of the image
        pos_inch[2] = embedding_image_size
        pos_inch[3] = embedding_image_size

        for j in range(rows):

            for i in range(4):

                # adds the plot to the figure
                ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

                if i == 1:
                    gap = inter_gap
                else:
                    gap = intra_gap

                # adds the inter plot gap
                pos_inch[0] += embedding_image_size + gap
                
            pos_inch[0] = 0

            if (j+1) % 2 == 0:
                pos_inch[1] -= embedding_image_size + inter_gap
            else:
                pos_inch[1] -= embedding_image_size + intra_gap

        # gets the DC voltage data - this is for only the on state or else it would all be 0
        voltage = self.dataset.dc_voltage

        # gets just part of the loop
        if hasattr(self.dataset, 'cycle') and self.dataset.cycle is not None:
            # gets the cycle of interest
            voltage = self.dataset.get_cycle(voltage)
            
        

        # # gets the index of the voltage steps to plot
        # inds = np.linspace(0,len(voltage)-1, number_of_steps, dtype=int)

        # # converts the data to a numpy array
        # if isinstance(SHO_, torch.Tensor):
        #     SHO_ = SHO_.detach().numpy()

        # SHO_ = SHO_.reshape(self.dataset.num_pix, self.dataset.voltage_steps, 4)

        # # get the selected measurement cycle
        # SHO_ = self.dataset.get_measurement_cycle(SHO_, axis = 1)
        
        ax_ = [ax[0]]
        
        if comparison is not None:
            z = 1
        else:
            z = 0
        
        for j in range(1 + z):
            for i in range(2):
                ax_.extend(ax[1+2*i+8*j:3+2*i+8*j])
                ax_.extend(ax[5+ 2*i + 8*j:7+2*i + 8*j])
                
        ax = ax_

        # plots the voltage
        ax[0].plot(voltage, "k")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].set_xlabel("Step")
        
        for i, ax in enumerate(ax):
            labelfigs(ax, i)
  

        # names = ["A", "\u03C9", "Q", "\u03C6"]

        # for i, ind in enumerate(inds):

        #     # loops around the amp, resonant frequency, and Q, Phase
        #     for j in range(4):
        #         imagemap(ax[i*4+j+1], SHO_[:, ind, j], colorbars=False, cmap="viridis",)

        #         if i // rows == 0:
        #             labelfigs(ax[i*4+j+1], string_add = names[j], loc = "cb", size = 5, inset_fraction=.2)

        #         ax[i*4+j+1].images[0].set_clim(clims[j])
        #     labelfigs(ax[1::4][i], string_add = str(i+1), size = 5, loc = "bl", inset_fraction=.2)

        # # if add colorbars
        # if colorbars:

        #     # builds a list to store the colorbar axis objects
        #     bar_ax = []

        #     #gets the voltage axis position in ([xmin, ymin, xmax, ymax]])
        #     voltage_ax_pos = fig_scalar.to_inches(np.array(ax[0].get_position()).flatten())

        #     # loops around the 4 axis
        #     for i in range(4):

        #         # calculates the height and width of the colorbars
        #         cbar_h = (voltage_ax_pos[1] - inter_gap - 2 * intra_gap - .33)/2
        #         cbar_w = (cbar_space - inter_gap - 2 * cbar_gap)/2

        #         # sets the position of the axis in inches
        #         pos_inch = [voltage_ax_pos[2] - (2 - i % 2)*(cbar_gap + cbar_w) + inter_gap,
        #                     voltage_ax_pos[1] - (i//2)*(inter_gap + cbar_h) -.33 - cbar_h,
        #                     cbar_w,
        #                     cbar_h]

        #         # adds the plot to the figure
        #         bar_ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

        #         # adds the colorbars to the plots
        #         cbar = plt.colorbar(ax[i+1].images[0], cax=bar_ax[i], format = '%.1e')
        #         cbar.set_label(names[i])  # Add a label to the colorbar

        # # prints the figure
        # if self.Printer is not None and filename is not None:
        #     self.Printer.savefig(fig, filename, label_figs=ax[1::4], size = 6, loc = 'tl', inset_fraction=.2)

        # fig.show()

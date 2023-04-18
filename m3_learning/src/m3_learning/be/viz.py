import numpy as np
from m3_learning.viz.layout import layout_fig
from scipy.signal import resample
from scipy import fftpack
import matplotlib.pyplot as plt


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
        ax_new.plot(np.repeat(dataset.hysteresis_waveform, 2))
        ax_new.set_xlim(x_start, x_end)
        ax_new.set_ylim(0, 15)

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
        ax[3].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
        ax2 = ax[3].twinx()
        ax2.plot(
            dataset.frequency_bin,
            data_[1].flatten(),
            "r",
        )
        ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

        dataset.raw_format = "complex"
        data_ = dataset.raw_spectra(pixel, voltagestep)

        # plots the real and imaginary components for the selected pixel and time step
        ax[4].plot(dataset.frequency_bin, data_[0].flatten(), label="Real")
        ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
        ax3 = ax[4].twinx()
        ax3.plot(
            dataset.frequency_bin, data_[1].flatten(), 'r', label="Imaginary")
        ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)")

        # prints the figure
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style='b')

    def SHO_hist(self, SHO_data, filename=""):

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
        if self.Printer is not None:
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

    def get_voltagestep(self, voltagestep):
        if voltagestep is None:

            if self.dataset.measurement_state == 'on' or self.dataset.measurement_state == 'off':
                voltagestep = np.random.randint(
                    0, self.dataset.voltage_steps // 2)
            else:
                voltagestep = np.random.randint(0, self.dataset.voltage_steps)
        return voltagestep

    def fit_tester(self, true, predict, pixel=None, voltagestep=None, **kwargs):

        # if a pixel is not provided it will select a random pixel
        if pixel is None:

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltagestep = self.get_voltagestep(voltagestep)

        params = self.dataset.SHO_LSQF(pixel=pixel, voltage_step=voltagestep)

        self.raw_data_comparison(
            true, predict, pixel=pixel, voltagestep=voltagestep, fit_results=params, **kwargs)

    def nn_checker(self, state, filename=None,
                   pixel=None, voltagestep=None, legend=True, **kwargs):

        # if a pixel is not provided it will select a random pixel
        if pixel is None:

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltagestep = self.get_voltagestep(voltagestep)

        self.set_attributes(**state)

        data = self.dataset.raw_spectra(pixel=pixel, voltage_step=voltagestep)

        # plot real and imaginary components of resampled data
        fig = plt.figure(figsize=(3, 1.25), layout='compressed')
        axs = plt.subplot(111)

        self.dataset.raw_format = "complex"

        x, data = self._get_data(pixel, voltagestep, **kwargs)

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

    def _get_data(self, pixel, voltagestep, **kwargs):

        data = self.dataset.raw_spectra(pixel=pixel,
                                        voltage_step=voltagestep,
                                        **kwargs)

        # get the correct frequency
        x = self.get_freq_values(data[0])
        return x, data

    def raw_data_comparison(self,
                            true,
                            predict=None,
                            filename=None,
                            pixel=None,
                            voltagestep=None,
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
        voltagestep = self.get_voltagestep(voltagestep)

        # sets the dataset state to grab the magnitude spectrum
        self.dataset.raw_format = "magnitude spectrum"

        x, data = self._get_data(pixel, voltagestep, **kwargs)

        axs[0].plot(x, data[0].flatten(), 'b',
                    label=self.dataset.label + " Amplitude")
        ax1 = axs[0].twinx()
        ax1.plot(x, data[1].flatten(), 'r',
                 label=self.dataset.label + " Phase")

        if predict is not None:
            self.set_attributes(**predict)
            x, data = self._get_data(pixel, voltagestep)
            axs[0].plot(x, data[0].flatten(), 'bo',
                        label=self.dataset.label + " Amplitude")
            ax1.plot(x, data[1].flatten(), 'ro',
                     label=self.dataset.label + " Phase")
            self.set_attributes(**true)

        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Amplitude (Arb. U.)")
        ax1.set_ylabel("Phase (rad)")

        self.dataset.raw_format = "complex"

        x, data = self._get_data(pixel, voltagestep, **kwargs)

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
            x, data = self._get_data(pixel, voltagestep)
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


# Class BE_Viz:

#     def __init__(self, dataset, shift=None, **kwargs):

#         self.dataset = dataset
#         self.shift = shift


# class Viz:

#        def __init__(self, dataset, state='lsqf', shift=None):

#             self.shift = shift

#             self.dataset = dataset
#             self.state = state
#             self.printing = self.dataset.printing

#             self.labels = [{'title': "Amplitude",
#                             'y_label': "Amplitude (Arb. U.)",
#                             'attr': "SHO_fit_amp"},
#                            {'title': "Resonance Frequency",
#                             'y_label': "Resonance Frequency (Hz)",
#                             'attr': "SHO_fit_resonance"},
#                            {'title': "Dampening",
#                             'y_label': "Quality Factor (Arb. U.)",
#                             'attr': "SHO_fit_q"},
#                            {'title': "Phase",
#                             'y_label': "Phase (rad)",
#                             'attr': "SHO_fit_phase"}]

#         def raw_be(self, filename="Figure_1_random_cantilever_resonance_results"):

#             # Select a random point and time step to plot
#             pixel = np.random.randint(0, self.dataset.num_pix)
#             voltagestep = np.random.randint(self.dataset.voltage_steps)

#             # prints the pixel and time step
#             print(pixel, voltagestep)

#             # Plots the amplitude and phase for the selected pixel and time step
#             fig, ax = layout_fig(5, 5, figsize=(6 * 11.2, 10))

#             # constructs the BE waveform and plot
#             be_voltagesteps = len(self.dataset.be_waveform) / \
#                 self.dataset.be_repeats

#             # plots the BE waveform
#             ax[0].plot(self.dataset.be_waveform[: int(be_voltagesteps)])
#             ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")
#             ax[0].set_title("BE Waveform")

#             # plots the resonance graph
#             resonance_graph = np.fft.fft(
#                 self.dataset.be_waveform[: int(be_voltagesteps)])
#             fftfreq = fftpack.fftfreq(int(be_voltagesteps)) * \
#                 self.dataset.sampling_rate
#             ax[1].plot(
#                 fftfreq[: int(be_voltagesteps) //
#                         2], np.abs(resonance_graph[: int(be_voltagesteps) // 2])
#             )
#             ax[1].axvline(
#                 x=self.dataset.be_center_frequency,
#                 ymax=np.max(resonance_graph[: int(be_voltagesteps) // 2]),
#                 linestyle="--",
#                 color="r",
#             )
#             ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
#             ax[1].set_xlim(
#                 self.dataset.be_center_frequency - self.dataset.be_bandwidth -
#                 self.dataset.be_bandwidth * 0.25,
#                 self.dataset.be_center_frequency + self.dataset.be_bandwidth +
#                 self.dataset.be_bandwidth * 0.25,
#             )

#             # manually set the x limits
#             x_start = 120
#             x_end = 140

#             # plots the hysteresis waveform and zooms in
#             ax[2].plot(self.dataset.hysteresis_waveform)
#             ax_new = fig.add_axes([0.52, 0.6, 0.3/5.5, 0.25])
#             ax_new.plot(np.repeat(self.dataset.hysteresis_waveform, 2))
#             ax_new.set_xlim(x_start, x_end)
#             ax_new.set_ylim(0, 15)
#             ax_new.set_xticks(np.linspace(x_start, x_end, 6))
#             ax_new.set_xticklabels([60, 62, 64, 66, 68, 70])
#             fig.add_artist(
#                 ConnectionPatch(
#                     xyA=(x_start // 2,
#                          self.dataset.hysteresis_waveform[x_start // 2]),
#                     coordsA=ax[2].transData,
#                     xyB=(105, 16),
#                     coordsB=ax[2].transData,
#                     color="green",
#                 )
#             )
#             fig.add_artist(
#                 ConnectionPatch(
#                     xyA=(x_end // 2,
#                          self.dataset.hysteresis_waveform[x_end // 2]),
#                     coordsA=ax[2].transData,
#                     xyB=(105, 4.5),
#                     coordsB=ax[2].transData,
#                     color="green",
#                 )
#             )
#             ax[2].set_xlabel("Voltage Steps")
#             ax[2].set_ylabel("Voltage (V)")

#             # plots the magnitude spectrum for and phase for the selected pixel and time step
#             ax[3].plot(
#                 original_x,
#                 self.dataset.get_spectra(
#                     self.dataset.magnitude_spectrum_amplitude, pixel, voltagestep),
#             )
#             ax[3].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
#             ax2 = ax[3].twinx()
#             ax2.plot(
#                 original_x,
#                 self.dataset.get_spectra(
#                     self.dataset.magnitude_spectrum_phase, pixel, voltagestep),
#                 "r+",
#             )
#             ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

#             # plots the real and imaginary components for the selected pixel and time step
#             ax[4].plot(original_x, self.dataset.get_spectra(
#                 self.dataset.complex_spectrum_real, pixel, voltagestep), label="Real")
#             ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
#             ax3 = ax[4].twinx()
#             ax3.plot(
#                 original_x, self.dataset.get_spectra(
#                     self.dataset.complex_spectrum_imag, pixel, voltagestep), 'r', label="Imaginary")
#             ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)")

#             # saves the figure
#             self.printing.savefig(
#                 fig, filename, tight_layout=False)

#         def SHO_hist(self, filename="Figure_3_SHO_fit_results_before_scaling", data_type=None):

#             if data_type == 'scaled':
#                 postfix = '_scaled'
#             else:
#                 postfix = ''

#             # check distributions of each parameter before and after scaling
#             fig, axs = layout_fig(4, 4, figsize=(20, 4))

#             for ax, label in zip(axs.flat, self.labels):
#                 data = getattr(self.dataset, label['attr'] + postfix)
#                 if label['attr'] == "SHO_fit_phase" and self.shift is not None and postfix == "":
#                     data = self.shift_phase(data)

#                 ax.hist(data.flatten(), 100)
#                 ax.set(xlabel=label['y_label'], ylabel="counts")
#                 ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

#             plt.tight_layout()

#             self.printing.savefig(fig, filename)

#         def SHO_loops(self, pix=None, filename="Figure_2_random_SHO_fit_results"):
#             if pix is None:
#                 # selects a random pixel to plot
#                 pix = np.random.randint(0, 3600)

#             # plots the SHO fit results for the selected pixel
#             fig, ax = layout_fig(4, 4, figsize=(30, 6))

#             for ax, label in zip(ax, self.labels):

#                 data = getattr(
#                     self.dataset, label['attr'])[pix, :]

#                 if label['attr'] == "SHO_fit_phase" and self.shift is not None:
#                     data = self.shift_phase(data)

#                 ax.plot(self.dataset.dc_voltage, data)
#                 ax.set_title(label['title'])
#                 ax.set_ylabel(label['y_label'])

#             fig.tight_layout()
#             self.printing.savefig(fig, filename)

#         def shift_phase(self, phase, shift_=None):

#             if shift_ is None:
#                 shift = self.shift
#             else:
#                 shift = shift_

#             phase_ = phase.copy()
#             phase_ += np.pi
#             phase_[phase_ <= shift] += 2 *\
#                 np.pi  # shift phase values greater than pi
#             return phase_ - shift - np.pi

#         def raw_data(self,
#                      original,
#                      predict,
#                      predict_label=None,
#                      filename=None):

#             if predict_label is not None:
#                 predict_label = ' ' + predict_label

#             if len(original) == len(self.dataset.wvec_freq):
#                 original_x = self.dataset.wvec_freq
#             elif len(original) == len(original_x):
#                 original_x = self.dataset.frequency_bins
#             else:
#                 raise ValueError(
#                     "original data must be the same length as the frequency bins or the resampled frequency bins")

#             # plot real and imaginary components of resampled data
#             fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

#             def plot_curve(axs, x, y, label, color, key=''):
#                 axs.plot(
#                     x,
#                     y,
#                     key,
#                     label=label,
#                     color=color,
#                 )

#             plot_curve(axs[0], original_x,
#                        np.abs(original),
#                        "amplitude", 'b')

#             plot_curve(axs[0], self.dataset.wvec_freq,
#                        np.abs(predict),
#                        f"amplitude {predict_label}", 'b', key='o')

#             axs[0].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

#             ax2 = axs[0].twinx()

#             plot_curve(ax2, original_x,
#                        np.angle(original),
#                        label="phase", color='r', key='s')

#             plot_curve(ax2, self.dataset.wvec_freq,
#                        np.angle(predict),
#                        label=f"phase {predict_label}", color='r')

#             ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

#             plot_curve(axs[1], original_x,
#                        np.real(original),
#                        "real", 'b', key='o')

#             plot_curve(axs[1], self.dataset.wvec_freq,
#                        np.real(predict),
#                        f"real {predict_label}", 'b')

#             axs[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

#             ax3 = axs[1].twinx()

#             plot_curve(ax3, original_x,
#                        np.imag(original),
#                        label="imaginary",
#                        color='r', key='s')

#             plot_curve(ax3, self.dataset.wvec_freq,
#                        np.imag(predict),
#                        label=f"imaginary {predict_label}", color='r')

#             ax3.set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

#             fig.legend(bbox_to_anchor=(1.16, 0.93),
#                        loc="upper right", borderaxespad=0.0)
#             if filename is not None:
#                 self.dataset.printing.savefig(fig, filename)

#         def raw_resampled_data(self, filename="Figure_4_raw_and_resampled_raw_data"):

#             # Select a random point and time step to plot
#             pixel = np.random.randint(0, self.dataset.num_pix)
#             voltagestep = np.random.randint(self.dataset.voltage_steps)

#             self.raw_data(self.dataset.raw_data.reshape(self.dataset.num_pix, -1, self.dataset.num_bins)[pixel, voltagestep],
#                           self.dataset.raw_data_resampled[pixel, voltagestep],
#                           predict_label=' resampled',
#                           filename=filename)

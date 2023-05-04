import numpy as np
from m3_learning.viz.layout import layout_fig, inset_connector, add_box, subfigures, add_text_to_figure
from scipy.signal import resample
from scipy import fftpack
import matplotlib.pyplot as plt
from m3_learning.be.nn import SHO_Model


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

    def get_voltagestep(self, voltage_step):
        if voltage_step is None:

            if self.dataset.measurement_state == 'on' or self.dataset.measurement_state == 'off':
                voltage_step = np.random.randint(
                    0, self.dataset.voltage_steps // 2)
            else:
                voltage_step = np.random.randint(0, self.dataset.voltage_steps)
        return voltage_step

    def fit_tester(self, true, predict, pixel=None, voltage_step=None, **kwargs):

        # if a pixel is not provided it will select a random pixel
        if pixel is None:

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltage_step = self.get_voltagestep(voltage_step)

        params = self.dataset.SHO_LSQF(pixel=pixel, voltage_step=voltage_step)

        self.raw_data_comparison(
            true, predict, pixel=pixel, voltage_step=voltage_step, fit_results=params, **kwargs)

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

    # def _get_data(self, pixel, voltagestep, **kwargs):

    #     data = self.dataset.raw_spectra(pixel=pixel,
    #                                     voltage_step=voltagestep,
    #                                     **kwargs)

    #     # get the correct frequency
    #     x = self.get_freq_values(data[0])
    #     return x, data

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
            pixel, voltage_step, frequency=True, **kwargs)

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
            pixel, voltage_step, frequency=True, **kwargs)

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

    def get_best_median_worst(self, 
                              true_state,
                              prediction=None, 
                            model=None,
                              out_state=None,
                              n=1, **kwargs):

        if type(true_state) is dict:

            self.set_attributes(**true_state)

            # the data must be scaled to rank the results
            self.dataset.scaled = True

            true, x1 = self.dataset.raw_spectra(frequency=True)
        
        # condition if x_data is passed
        else:
            
            # converts to a standard form which is a list 
            true = self.dataset.to_real_imag(true_state)
            
            # converts to numpy from tensor
            true = [data.numpy() for data in true]
            
            # gets the frequency values
            if true[0].ndim == 2:
                x1 = self.dataset.get_freq_values(true[0].shape[1])
                                
        # holds the raw state
        current_state = self.dataset.get_state

        if model is not None:

            # sets the phase shift to zero for parameters
            # This is important if doing the fits because the fits will be wrong if the phase is shifted.
            self.dataset.NN_phase_shift = 0

            data = self.dataset.to_nn(true)

            pred_data, scaled_params, params = model.predict(data)

            prediction, x2 = self.dataset.raw_spectra(
                fit_results=params, frequency=True)      
            
        # this must take the scaled data
        index1, mse1, d1, d2 = SHO_Model.get_rankings(true, prediction, n=n)

        def convert_to_mag(data):
            data = self.dataset.to_complex(data)
            data = self.dataset.raw_data_scaler.inverse_transform(data)
            return self.dataset.to_magnitude(data)

        labels = ["real", "imaginary"]

        if out_state is not None:
            if "measurement state" in out_state.keys():
                if out_state["measurement_state"] == "magnitude spectrum":
                    d1 = convert_to_mag(d1)
                    d2 = convert_to_mag(d2)
                    labels = ["Amplitude", "Phase"]
            elif "scaled" in out_state.keys():
                if out_state["scaled"] == False:
                    d1 = self.dataset.raw_data_scaler.inverse_transform(d1)
                    d2 = self.dataset.raw_data_scaler.inverse_transform(d2)
                    labels = ["Scaled " + s for s in labels]
                    
        self.set_attributes(**current_state)

        return d1, d2, x1, x2, labels, index1, mse1

    def bmw_nn(self, true_state,
               prediction=None,
               model=None,
               out_state=None,
               n=1, 
               gaps = (.8,.33),
               size = (1.25,1.25),
               filename = None, 
               **kwargs):
 
        
        d1, d2, x1, x2, label, index1, mse1 = self.get_best_median_worst(
                                            true_state,
                                            prediction=prediction, 
                                            model=model,
                                            out_state=out_state,
                                            n=n,
                                            **kwargs)

        print(d1.shape, d2.shape)

        fig, ax = subfigures(1, 3, gaps=gaps, size=size)
        
        ax.reverse()
        
        for i, (true, prediction, error) in enumerate(zip(d1, d2, mse1)):

            ax_ = ax[i]
            ax_.plot(x2, prediction[0].flatten(), 'b',
                     label=f"NN {label[0]}")
            ax1 = ax_.twinx()
            ax1.plot(x2, prediction[1].flatten(), 'r',
                     label=f"NN {label[1]}]")

            ax_.plot(x1, true[0].flatten(), 'bo',
                     label=f"Raw {label[0]}")
            ax1.plot(x1, true[1].flatten(), 'ro',
                     label=f"Raw {label[1]}")

            ax_.set_xlabel("Frequency (Hz)")
            
            
            # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
            text_position_in_inches = (
                -1*(gaps[0] + size[0])*((2-i) % 3) + size[0]/2, (gaps[1] + size[1])*(1.25-i//3-1.25) - gaps[1])
            text = f'MSE: {error:0.4f}'
            add_text_to_figure(fig, text, text_position_in_inches, fontsize=6, ha='center')

            if "measurement state" in out_state.keys():
                if out_state["measurement_state"] == "magnitude spectrum":
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
            self.Printer.savefig(fig, filename, style='b')
                

        if "returns" in kwargs.keys():
            if kwargs["returns"] == True:
                return d1, d2, index1, mse1


#     def best_median_worst_fit_comparison(self):

#         # for the SHO curves it makes sense to determine the error based on the normalized fit results in complex form.
#         state = {'fitter': 'LSQF',
#                  'resampled': False,
#                  'scaled': True,
#                  "raw_format": "complex", }

#         self.set_attributes(**state)

#         fit_results_compare = self.dataset.raw_spectra(
#             fit_results=self.dataset.SHO_fit_results())

#         raw_SHO = self.dataset.raw_spectra()

#         index1, mse1, d1, d2 = SHO_Model.get_rankings(
#             raw_SHO, fit_results_compare, n=1)

#         d1 = np.swapaxes(d1, 1, 2)
#         d2 = np.swapaxes(d2, 1, 2)

#         d1 = self.dataset.raw_data_scaler.inverse_transform(d1)
#         d2 = self.dataset.raw_data_scaler.inverse_transform(d2)

#         d1 = np.array(self.dataset.to_magnitude(d1))
#         d2 = np.array(self.dataset.to_magnitude(d2))

#         d1 = np.swapaxes(d1, 0, 1)
#         d2 = np.swapaxes(d2, 0, 1)

#         fig, ax = subfigures(3, 2, gaps=(.8, .9), size=(1.25, 1.25))

#         x = self.get_freq_values(d1[0][1])

#         ax.reverse()

#         for i, (true, prediction) in enumerate(zip(d1, d2)):
#             print(mse1[i])
#             ax_ = ax[i*2]
#             ax_.plot(x, prediction[0].flatten(), 'b',
#                      label="LSQF Amplitude")
#             ax1 = ax_.twinx()
#             ax1.plot(x, prediction[1].flatten(), 'r',
#                      label="LSQF Phase")

#             ax_.plot(x, true[0].flatten(), 'bo',
#                      label="Raw Amplitude")
#             ax1.plot(x, true[1].flatten(), 'ro',
#                      label="Raw Phase")

#             ax_.set_xlabel("Frequency (Hz)")
#             ax_.set_ylabel("Amplitude (Arb. U.)")
#             ax1.set_ylabel("Phase (rad)")


# # Class BE_Viz:

# #     def __init__(self, dataset, shift=None, **kwargs):

# #         self.dataset = dataset
# #         self.shift = shift


# # class Viz:

# #        def __init__(self, dataset, state='lsqf', shift=None):

# #             self.shift = shift

# #             self.dataset = dataset
# #             self.state = state
# #             self.printing = self.dataset.printing

# #             self.labels = [{'title': "Amplitude",
# #                             'y_label': "Amplitude (Arb. U.)",
# #                             'attr': "SHO_fit_amp"},
# #                            {'title': "Resonance Frequency",
# #                             'y_label': "Resonance Frequency (Hz)",
# #                             'attr': "SHO_fit_resonance"},
# #                            {'title': "Dampening",
# #                             'y_label': "Quality Factor (Arb. U.)",
# #                             'attr': "SHO_fit_q"},
# #                            {'title': "Phase",
# #                             'y_label': "Phase (rad)",
# #                             'attr': "SHO_fit_phase"}]

# #         def raw_be(self, filename="Figure_1_random_cantilever_resonance_results"):

# #             # Select a random point and time step to plot
# #             pixel = np.random.randint(0, self.dataset.num_pix)
# #             voltagestep = np.random.randint(self.dataset.voltage_steps)

# #             # prints the pixel and time step
# #             print(pixel, voltagestep)

# #             # Plots the amplitude and phase for the selected pixel and time step
# #             fig, ax = layout_fig(5, 5, figsize=(6 * 11.2, 10))

# #             # constructs the BE waveform and plot
# #             be_voltagesteps = len(self.dataset.be_waveform) / \
# #                 self.dataset.be_repeats

# #             # plots the BE waveform
# #             ax[0].plot(self.dataset.be_waveform[: int(be_voltagesteps)])
# #             ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")
# #             ax[0].set_title("BE Waveform")

# #             # plots the resonance graph
# #             resonance_graph = np.fft.fft(
# #                 self.dataset.be_waveform[: int(be_voltagesteps)])
# #             fftfreq = fftpack.fftfreq(int(be_voltagesteps)) * \
# #                 self.dataset.sampling_rate
# #             ax[1].plot(
# #                 fftfreq[: int(be_voltagesteps) //
# #                         2], np.abs(resonance_graph[: int(be_voltagesteps) // 2])
# #             )
# #             ax[1].axvline(
# #                 x=self.dataset.be_center_frequency,
# #                 ymax=np.max(resonance_graph[: int(be_voltagesteps) // 2]),
# #                 linestyle="--",
# #                 color="r",
# #             )
# #             ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
# #             ax[1].set_xlim(
# #                 self.dataset.be_center_frequency - self.dataset.be_bandwidth -
# #                 self.dataset.be_bandwidth * 0.25,
# #                 self.dataset.be_center_frequency + self.dataset.be_bandwidth +
# #                 self.dataset.be_bandwidth * 0.25,
# #             )

# #             # manually set the x limits
# #             x_start = 120
# #             x_end = 140

# #             # plots the hysteresis waveform and zooms in
# #             ax[2].plot(self.dataset.hysteresis_waveform)
# #             ax_new = fig.add_axes([0.52, 0.6, 0.3/5.5, 0.25])
# #             ax_new.plot(np.repeat(self.dataset.hysteresis_waveform, 2))
# #             ax_new.set_xlim(x_start, x_end)
# #             ax_new.set_ylim(0, 15)
# #             ax_new.set_xticks(np.linspace(x_start, x_end, 6))
# #             ax_new.set_xticklabels([60, 62, 64, 66, 68, 70])
# #             fig.add_artist(
# #                 ConnectionPatch(
# #                     xyA=(x_start // 2,
# #                          self.dataset.hysteresis_waveform[x_start // 2]),
# #                     coordsA=ax[2].transData,
# #                     xyB=(105, 16),
# #                     coordsB=ax[2].transData,
# #                     color="green",
# #                 )
# #             )
# #             fig.add_artist(
# #                 ConnectionPatch(
# #                     xyA=(x_end // 2,
# #                          self.dataset.hysteresis_waveform[x_end // 2]),
# #                     coordsA=ax[2].transData,
# #                     xyB=(105, 4.5),
# #                     coordsB=ax[2].transData,
# #                     color="green",
# #                 )
# #             )
# #             ax[2].set_xlabel("Voltage Steps")
# #             ax[2].set_ylabel("Voltage (V)")

# #             # plots the magnitude spectrum for and phase for the selected pixel and time step
# #             ax[3].plot(
# #                 original_x,
# #                 self.dataset.get_spectra(
# #                     self.dataset.magnitude_spectrum_amplitude, pixel, voltagestep),
# #             )
# #             ax[3].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
# #             ax2 = ax[3].twinx()
# #             ax2.plot(
# #                 original_x,
# #                 self.dataset.get_spectra(
# #                     self.dataset.magnitude_spectrum_phase, pixel, voltagestep),
# #                 "r+",
# #             )
# #             ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

# #             # plots the real and imaginary components for the selected pixel and time step
# #             ax[4].plot(original_x, self.dataset.get_spectra(
# #                 self.dataset.complex_spectrum_real, pixel, voltagestep), label="Real")
# #             ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
# #             ax3 = ax[4].twinx()
# #             ax3.plot(
# #                 original_x, self.dataset.get_spectra(
# #                     self.dataset.complex_spectrum_imag, pixel, voltagestep), 'r', label="Imaginary")
# #             ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)")

# #             # saves the figure
# #             self.printing.savefig(
# #                 fig, filename, tight_layout=False)

# #         def SHO_hist(self, filename="Figure_3_SHO_fit_results_before_scaling", data_type=None):

# #             if data_type == 'scaled':
# #                 postfix = '_scaled'
# #             else:
# #                 postfix = ''

# #             # check distributions of each parameter before and after scaling
# #             fig, axs = layout_fig(4, 4, figsize=(20, 4))

# #             for ax, label in zip(axs.flat, self.labels):
# #                 data = getattr(self.dataset, label['attr'] + postfix)
# #                 if label['attr'] == "SHO_fit_phase" and self.shift is not None and postfix == "":
# #                     data = self.shift_phase(data)

# #                 ax.hist(data.flatten(), 100)
# #                 ax.set(xlabel=label['y_label'], ylabel="counts")
# #                 ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

# #             plt.tight_layout()

# #             self.printing.savefig(fig, filename)

# #         def SHO_loops(self, pix=None, filename="Figure_2_random_SHO_fit_results"):
# #             if pix is None:
# #                 # selects a random pixel to plot
# #                 pix = np.random.randint(0, 3600)

# #             # plots the SHO fit results for the selected pixel
# #             fig, ax = layout_fig(4, 4, figsize=(30, 6))

# #             for ax, label in zip(ax, self.labels):

# #                 data = getattr(
# #                     self.dataset, label['attr'])[pix, :]

# #                 if label['attr'] == "SHO_fit_phase" and self.shift is not None:
# #                     data = self.shift_phase(data)

# #                 ax.plot(self.dataset.dc_voltage, data)
# #                 ax.set_title(label['title'])
# #                 ax.set_ylabel(label['y_label'])

# #             fig.tight_layout()
# #             self.printing.savefig(fig, filename)

# #         def shift_phase(self, phase, shift_=None):

# #             if shift_ is None:
# #                 shift = self.shift
# #             else:
# #                 shift = shift_

# #             phase_ = phase.copy()
# #             phase_ += np.pi
# #             phase_[phase_ <= shift] += 2 *\
# #                 np.pi  # shift phase values greater than pi
# #             return phase_ - shift - np.pi

# #         def raw_data(self,
# #                      original,
# #                      predict,
# #                      predict_label=None,
# #                      filename=None):

# #             if predict_label is not None:
# #                 predict_label = ' ' + predict_label

# #             if len(original) == len(self.dataset.wvec_freq):
# #                 original_x = self.dataset.wvec_freq
# #             elif len(original) == len(original_x):
# #                 original_x = self.dataset.frequency_bins
# #             else:
# #                 raise ValueError(
# #                     "original data must be the same length as the frequency bins or the resampled frequency bins")

# #             # plot real and imaginary components of resampled data
# #             fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# #             def plot_curve(axs, x, y, label, color, key=''):
# #                 axs.plot(
# #                     x,
# #                     y,
# #                     key,
# #                     label=label,
# #                     color=color,
# #                 )

# #             plot_curve(axs[0], original_x,
# #                        np.abs(original),
# #                        "amplitude", 'b')

# #             plot_curve(axs[0], self.dataset.wvec_freq,
# #                        np.abs(predict),
# #                        f"amplitude {predict_label}", 'b', key='o')

# #             axs[0].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

# #             ax2 = axs[0].twinx()

# #             plot_curve(ax2, original_x,
# #                        np.angle(original),
# #                        label="phase", color='r', key='s')

# #             plot_curve(ax2, self.dataset.wvec_freq,
# #                        np.angle(predict),
# #                        label=f"phase {predict_label}", color='r')

# #             ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

# #             plot_curve(axs[1], original_x,
# #                        np.real(original),
# #                        "real", 'b', key='o')

# #             plot_curve(axs[1], self.dataset.wvec_freq,
# #                        np.real(predict),
# #                        f"real {predict_label}", 'b')

# #             axs[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

# #             ax3 = axs[1].twinx()

# #             plot_curve(ax3, original_x,
# #                        np.imag(original),
# #                        label="imaginary",
# #                        color='r', key='s')

# #             plot_curve(ax3, self.dataset.wvec_freq,
# #                        np.imag(predict),
# #                        label=f"imaginary {predict_label}", color='r')

# #             ax3.set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

# #             fig.legend(bbox_to_anchor=(1.16, 0.93),
# #                        loc="upper right", borderaxespad=0.0)
# #             if filename is not None:
# #                 self.dataset.printing.savefig(fig, filename)

# #         def raw_resampled_data(self, filename="Figure_4_raw_and_resampled_raw_data"):

# #             # Select a random point and time step to plot
# #             pixel = np.random.randint(0, self.dataset.num_pix)
# #             voltagestep = np.random.randint(self.dataset.voltage_steps)

# #             self.raw_data(self.dataset.raw_data.reshape(self.dataset.num_pix, -1, self.dataset.num_bins)[pixel, voltagestep],
# #                           self.dataset.raw_data_resampled[pixel, voltagestep],
# #                           predict_label=' resampled',
# #                           filename=filename)

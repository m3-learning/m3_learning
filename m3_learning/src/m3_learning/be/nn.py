import torch.nn as nn
import torch
from ..optimizers.AdaHessian import AdaHessian
from ..nn.random import random_seed
from ..nn.benchmarks.inference import computeTime
from ..viz.layout import get_axis_range, set_axis, Axis_Ratio
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import resample
from m3_learning.util.file_IO import make_folder


def SHO_fit_func_nn(params,
                    wvec_freq,
                    device='cpu'):
    """_summary_

    Returns:
        _type_: _description_
    """

    Amp = params[:, 0].type(torch.complex128)
    w_0 = params[:, 1].type(torch.complex128)
    Q = params[:, 2].type(torch.complex128)
    phi = params[:, 3].type(torch.complex128)
    wvec_freq = torch.tensor(wvec_freq)

    Amp = torch.unsqueeze(Amp, 1)
    w_0 = torch.unsqueeze(w_0, 1)
    phi = torch.unsqueeze(phi, 1)
    Q = torch.unsqueeze(Q, 1)

    wvec_freq = wvec_freq.to(device)

    numer = Amp * torch.exp((1.j) * phi) * torch.square(w_0)
    den_1 = torch.square(wvec_freq)
    den_2 = (1.j) * wvec_freq.to(device) * w_0 / Q
    den_3 = torch.square(w_0)

    den = den_1 - den_2 - den_3

    func = numer / den

    return func


class AE_Fitter_SHO(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
            nn.SELU(),
        )

        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
        )

        # 2nd block of 1d-conv layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
        )

        # Flatten layer
        self.flatten_layer = nn.Flatten()

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(26, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, 4),
        )

    def forward(self, x, n=-1):
        # output shape - samples, (real, imag), frequency
        x = torch.swapaxes(x, 1, 2)
        x = self.hidden_x1(x)
        xfc = torch.reshape(x, (n, 256))  # batch size, features
        xfc = self.hidden_xfc(xfc)

        # batch size, (real, imag), timesteps
        x = torch.reshape(x, (n, 2, 128))
        x = self.hidden_x2(x)
        cnn_flat = self.flatten_layer(x)
        encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
        embedding = self.hidden_embedding(encoded)  # output is 4 parameters

        # corrects the scaling of the parameters
        unscaled_param = (
            embedding *
            torch.tensor(self.dataset.SHO_scaler.var_ ** 0.5).cuda()
            + torch.tensor(self.dataset.SHO_scaler.mean_).cuda()
        )

        frequency_bins = resample(self.dataset.frequency_bin,
                                  self.dataset.resampled_bins)

        # passes to the pytorch fitting function
        fits = SHO_fit_func_nn(
            unscaled_param, frequency_bins, device=self.device)

        # extract and return real and imaginary
        real = torch.real(fits)
        real_scaled = (real - torch.tensor(self.dataset.raw_data_scaler.real_scaler.mean).cuda()) / torch.tensor(
            self.dataset.raw_data_scaler.real_scaler.std
        ).cuda()
        imag = torch.imag(fits)
        imag_scaled = (imag - torch.tensor(self.dataset.raw_data_scaler.imag_scaler.mean).cuda()) / torch.tensor(
            self.dataset.raw_data_scaler.imag_scaler.std
        ).cuda()
        out = torch.stack((real_scaled, imag_scaled), 2)
        if self.training == True:
            return out
        if self.training == False:
            return out, embedding, unscaled_param


class SHO_Model(AE_Fitter_SHO):

    def __init__(self,
                 dataset,
                 model_basename='',
                 training=True,
                 path='Trained Models/SHO Fitter/',
                 device=None):

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using GPU {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("Using CPU")

        super().__init__(self.device)

        self.model = AE_Fitter_SHO(self.device)
        self.model.dataset = dataset
        self.model.training = True
        self.model_name = model_basename
        self.path = make_folder(path)

    def fit(self,
            data_train,
            batch_size=200,
            epochs=5,
            loss_func=torch.nn.MSELoss(),
            optimizer='Adam',
            seed=42,
            datatype=torch.float32,
            save_all=False,
            **kwargs):

        # sets the model to be a specific datatype and on cuda
        self.to(datatype).to(self.device)

        # Note that the seed will behave differently on different hardware targets (GPUs)
        random_seed(seed=seed)

        torch.cuda.empty_cache()

        # selects the optimizer
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters())
        elif optimizer == "AdaHessian":
            optimizer = AdaHessian(self.model.parameters(), lr=0.1)

        # instantiate the dataloader
        train_dataloader = DataLoader(
            data_train, batch_size=batch_size, shuffle=True)

        # loops around each epoch
        for epoch in range(epochs):

            # starts the timer
            start_time = time.time()

            train_loss = 0.0
            total_num = 0

            # sets the model to training mode
            self.model.train()

            for train_batch in train_dataloader:

                train_batch = train_batch.to(datatype).to(self.device)

                pred = self.model(train_batch).to(torch.float32)

                optimizer.zero_grad()

                loss = loss_func(train_batch, pred).to(torch.float32)
                loss.backward(create_graph=True)
                train_loss += loss.item() * pred.shape[0]
                total_num += pred.shape[0]

                optimizer.step()

            train_loss /= total_num

            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
                                                              1, epochs, train_loss))
            print("--- %s seconds ---" % (time.time() - start_time))

            if save_all:
                torch.save(self.model.state_dict(),
                           f"{self.path}/{self.model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth")

        torch.save(self.model.state_dict(),
                   f"{self.path}/{self.model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth")


# def SHO_fit_func_nn(parms,
#                        wvec_freq,
#                        device='cpu'):

#     Amp = parms[:, 0].type(torch.complex128)
#     w_0 = parms[:, 1].type(torch.complex128)
#     Q = parms[:, 2].type(torch.complex128)
#     phi = parms[:, 3].type(torch.complex128)
#     wvec_freq = torch.tensor(wvec_freq)

#     Amp = torch.unsqueeze(Amp, 1)
#     w_0 = torch.unsqueeze(w_0, 1)
#     phi = torch.unsqueeze(phi, 1)
#     Q = torch.unsqueeze(Q, 1)

#     wvec_freq = wvec_freq.to(device)

#     numer = Amp * torch.exp((1.j) * phi) * torch.square(w_0)
#     den_1 = torch.square(wvec_freq)
#     den_2 = (1.j) * wvec_freq.to(device) * w_0 / Q
#     den_3 = torch.square(w_0)

#     den = den_1 - den_2 - den_3

#     func = numer / den

#     return func


# class SHO_Model(nn.Module):
#     def __init__(self, dataset, training=True):
#         super().__init__()
#         self.dataset = dataset
#         if ~hasattr(self.dataset, "SHO_scaler"):
#             self.dataset.SHO_Scaler()

#         self.training = training

#         # Input block of 1d convolution
#         self.hidden_x1 = nn.Sequential(
#             nn.Conv1d(in_channels=2, out_channels=8, kernel_size=7),
#             nn.SELU(),
#             nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
#             nn.SELU(),
#             nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
#             nn.SELU(),
#         )

#         # fully connected block
#         self.hidden_xfc = nn.Sequential(
#             nn.Linear(256, 20),
#             nn.SELU(),
#             nn.Linear(20, 20),
#             nn.SELU(),
#         )

#         # 2nd block of 1d-conv layers
#         self.hidden_x2 = nn.Sequential(
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5),
#             nn.SELU(),
#             nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
#             nn.SELU(),
#             nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
#             nn.SELU(),
#             nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
#             nn.SELU(),
#             nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
#             nn.SELU(),
#             nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
#             nn.SELU(),
#             nn.AvgPool1d(kernel_size=2),
#             nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
#             nn.SELU(),
#             nn.AvgPool1d(kernel_size=2),
#             nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
#             nn.SELU(),
#             nn.AvgPool1d(kernel_size=2),
#         )

#         # Flatten layer
#         self.flatten_layer = nn.Flatten()

#         # Final embedding block - Output 4 values - linear
#         self.hidden_embedding = nn.Sequential(
#             nn.Linear(26, 16),
#             nn.SELU(),
#             nn.Linear(16, 8),
#             nn.SELU(),
#             nn.Linear(8, 4),
#         )

#     # @property
#     # def training(self):
#     #     return self._training

#     # @training.setter
#     # def train(self, value):
#     #     self._training = value

#     def forward(self, x, n=-1):
#         # output shape - samples, (real, imag), frequency
#         x = torch.swapaxes(x, 1, 2)
#         x = self.hidden_x1(x)
#         xfc = torch.reshape(x, (n, 256))  # batch size, features
#         xfc = self.hidden_xfc(xfc)

#         # batch size, (real, imag), timesteps
#         x = torch.reshape(x, (n, 2, 128))
#         x = self.hidden_x2(x)
#         cnn_flat = self.flatten_layer(x)
#         encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
#         embedding = self.hidden_embedding(encoded)  # output is 4 parameters

#         # corrects the scaling of the parameters
#         unscaled_param = (
#             embedding *
#             torch.tensor(self.dataset.SHO_scaler.var_ ** 0.5).cuda()
#             + torch.tensor(self.dataset.SHO_scaler.mean_).cuda()
#         )

#         frequency_bins = resample(self.dataset.frequency_bin,
#                                   self.dataset.resampled_bins)

#         # passes to the pytorch fitting function
#         fits = SHO_fit_func_nn(
#             unscaled_param, frequency_bins, device="cuda")

#         # extract and return real and imaginary
#         real = torch.real(fits)
#         real_scaled = (real - torch.tensor(self.dataset.raw_data_scalar.real_scaler.mean).cuda()) / torch.tensor(
#             self.dataset.raw_data_scalar.real_scaler.std
#         ).cuda()
#         imag = torch.imag(fits)
#         imag_scaled = (imag - torch.tensor(self.dataset.raw_data_scalar.imag_scaler.mean).cuda()) / torch.tensor(
#             self.dataset.raw_data_scalar.imag_scaler.std
#         ).cuda()
#         out = torch.stack((real_scaled, imag_scaled), 2)
#         if self.training == True:
#             return out
#         if self.training == False:
#             return out, embedding, unscaled_param


# class SHO_NN_Model:

#     def __init__(self,
#                  model,
#                  seed=42,
#                  lr=0.1,
#                  **kwargs):
#         super().__init__()
#         self.model = model
#         self.seed = seed
#         self.lr = lr
#         self.__dict__.update(kwargs)

#     def train(self,
#               data_train,
#               batch_size,
#               epochs=5,
#               loss_func=torch.nn.MSELoss(),
#               optimizer='Adam',
#               **kwargs):

#         # Note that the seed will behave differently on different hardware targets (GPUs)
#         random_seed(seed=self.seed)

#         torch.cuda.empty_cache()

#         # selects the optimizer
#         if optimizer == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters())
#         elif optimizer == "AdaHessian":
#             optimizer = AdaHessian(self.model.parameters(), lr=0.1)

#         # instantiate the dataloader
#         train_dataloader = DataLoader(
#             data_train, batch_size=batch_size, shuffle=True)

#         for epoch in range(epochs):
#             start_time = time.time()

#             train_loss = 0.0
#             total_num = 0

#             self.model.train()

#             for train_batch in train_dataloader:

#                 pred = self.model(train_batch.double().cuda())

#                 optimizer.zero_grad()

#                 loss = loss_func(train_batch.double().cuda(), pred)
#                 loss.backward(create_graph=True)
#                 train_loss += loss.item() * pred.shape[0]
#                 total_num += pred.shape[0]

#                 optimizer.step()

#             train_loss /= total_num

#             print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
#                   1, epochs, train_loss))
#             print("--- %s seconds ---" % (time.time() - start_time))

#             torch.save(self.model.state_dict(),
#                        'Trained Models/SHO Fitter/model.pth')

#     def inference_calculator(self, data, batch_size=.5e4):
#         torch.cuda.empty_cache()

#         batch_size = int(batch_size)

#         dataloader = DataLoader(data, batch_size)

#         # Computes the inference time
#         computeTime(self.model, next(iter(dataloader)).double(), batch_size)

#     def predict(self, data, validation, batch_size=10000):

#         dataloader = DataLoader(data, batch_size=batch_size)

#         # preallocate the predictions
#         num_elements = len(dataloader.dataset)
#         num_batches = len(dataloader)
#         predictions = torch.zeros_like(torch.tensor(data))
#         params_scaled = torch.zeros((data.shape[0], 4))
#         params = torch.zeros((data.shape[0], 4))

#         # compute the predictions
#         for i, train_batch in enumerate(dataloader):
#             start = i * batch_size
#             end = start + batch_size

#             if i == num_batches - 1:
#                 end = num_elements

#             pred_batch, params_scaled_, params_ = self.model(
#                 train_batch.double().cuda())

#             predictions[start:end] = pred_batch.cpu().detach()
#             params_scaled[start:end] = params_scaled_.cpu().detach()
#             params[start:end] = params_.cpu().detach()

#             torch.cuda.empty_cache()

#         if validation:
#             name = "validation"
#         else:
#             name = "predictions"

#         exec(f"self.model.dataset.nn_{name} = predictions")
#         exec(f"self.model.dataset.nn_{name}_params_scaled=params_scaled")
#         exec(f"self.model.dataset.nn_{name}_params =params")

#     def unscale_complex(self, data):

#         unscaled = np.zeros(data.shape)

#         # unscale the test data and predictions
#         unscaled[:, :, 0] = self.model.dataset.real_scaler.inverse_transform(
#             data[:, :, 0])
#         unscaled[:, :, 1] = self.model.dataset.imag_scaler.inverse_transform(
#             data[:, :, 1])
#         return unscaled[:, :, 0] + 1j * unscaled[:, :, 1]

#     def SHO_best_and_worst(self, true, predictions):

#         # computes the MSE for the real and imaginary components
#         mse_real = mean_squared_error(
#             true[:, :, 0], predictions[:, :, 0]
#         )
#         mse_imag = mean_squared_error(
#             true[:, :, 1], predictions[:, :, 1]
#         )
#         print(f"MSE for real component: {mse_real}")
#         print(f"MSE for imaginary component: {mse_imag}")

#         # computes the average MSE
#         error = (mse_real + mse_imag) / 2.0
#         print(f"Average MSE: {error}")

#         errors = np.sum(
#             np.mean(np.square(true - predictions), 1), 1
#         )
#         errors = np.asarray(errors)

#         return errors

#     def best_and_worst(self, true, prediction, filename="Figure_8_5_worst_and_best_pytorch_fit"):

#         def plot_curve(axs, x, y, label, color, key=''):
#             axs.plot(
#                 x,
#                 y,
#                 key,
#                 label=label,
#                 color=color,
#             )

#         errors = self.SHO_best_and_worst(true, prediction)

#         # sorting by highest and lowest MSE
#         best = (-errors).argsort()[:5]
#         # sorting by highest and lowest MSE
#         worst = (-errors).argsort()[-5:]

#         data = [best, worst]

#         # plotting the 5 worst and best reconstructions
#         fig, axs = plt.subplots(2, 5, figsize=(5.5, 2))

#         for j, list_ in enumerate(data):
#             ax1 = []
#             for i, ind in enumerate(list_):

#                 original_ = self.unscale_complex(true[[ind], :])[0]
#                 predicted_ = self.unscale_complex(prediction[[ind], :])[0]

#                 if len(original_) == len(self.model.dataset.wvec_freq):
#                     original_x = self.model.dataset.wvec_freq
#                 elif len(original_) == len(original_x):
#                     original_x = self.model.dataset.frequency_bins
#                 else:
#                     raise ValueError(
#                         "original data must be the same length as the frequency bins or the resampled frequency bins")

#                 plot_curve(axs[j, i], original_x, np.abs(original_),
#                            label="Raw Magnitude", color='b', key='o')

#                 plot_curve(axs[j, i], original_x, np.abs(predicted_),
#                            label="Raw Magnitude", color='b')

#                 ax1.append(axs[j, i].twinx())

#                 plot_curve(ax1[i], original_x, np.angle(original_),
#                            label="Raw Phase", color='r', key='o')

#                 plot_curve(ax1[i], original_x, np.angle(predicted_),
#                            label="Raw Phase", color='r')

#                 if i > 0:
#                     axs[j, i].set_yticklabels([])
#                 else:
#                     axs[j, i].set_ylabel('Magnitude')

#                 if i < 4:
#                     ax1[i].set_yticklabels([])
#                 else:
#                     ax1[i].set_ylabel('Phase')

#                 if j == 1:
#                     axs[j, i].set_xlabel('Frequency (Hz)')

#         set_axis(axs[j, :], get_axis_range(axs[j, :]))

#         set_axis(ax1, get_axis_range(ax1))

#         self.model.dataset.printing.savefig(fig, filename, tight_layout=False)

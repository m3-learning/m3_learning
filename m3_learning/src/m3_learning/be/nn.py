import torch.nn as nn
import torch
from ..optimizers.AdaHessian import AdaHessian
from ..nn.random import random_seed
from m3_learning.nn.benchmarks.inference import computeTime
from ..viz.layout import get_axis_range, set_axis, Axis_Ratio
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import resample
from m3_learning.util.file_IO import make_folder, append_to_csv
import itertools
from m3_learning.optimizers.TrustRegion import TrustRegion


def static_state_decorator(func):
    """Decorator that stops the function from changing the state

    Args:
        func (method): any method
    """
    def wrapper(*args, **kwargs):
        current_state = args[0].get_state
        out = func(*args, **kwargs)
        args[0].set_attributes(**current_state)
        return out
    return wrapper


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
            nn.AdaptiveAvgPool1d(64)
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
            nn.AdaptiveAvgPool1d(16),  # Adaptive pooling layer
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(8),  # Adaptive pooling layer
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(4),  # Adaptive pooling layer
        )

        # Flatten layer
        self.flatten_layer = nn.Flatten()

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(28, 16),
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

        # unscaled_param[:,0] = torch.nn.functional.relu(unscaled_param[:, 0])

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
            return out, unscaled_param
        if self.training == False:
            # this is a scaling that includes the corrections for shifts in the data
            embeddings = (unscaled_param.cuda() - torch.tensor(self.dataset.SHO_scaler.mean_).cuda()
                          )/torch.tensor(self.dataset.SHO_scaler.var_ ** 0.5).cuda()
            return out, embeddings, unscaled_param


class SHO_Model(AE_Fitter_SHO):

    def __init__(self,
                 dataset,
                 model_basename='',
                 training=True,
                 path='Trained Models/SHO Fitter/',
                 device=None,
                 **kwargs):

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
            write_CSV=None,
            closure=None,
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
            optimizer = AdaHessian(self.model.parameters(), lr=.5)
        elif isinstance(optimizer, dict):
            if optimizer['name'] == "TRCG":
                optimizer = optimizer['optimizer'](
                    self.model, optimizer['radius'], optimizer['device'])
        elif isinstance(optimizer, dict):
            if optimizer['name'] == "TRCG":
                optimizer_ = optimizer['optimizer'](
                    self.model, optimizer['radius'], optimizer['device'])
        else:
            try:
                optimizer = optimizer(self.model.parameters())
            except:
                raise ValueError("Optimizer not recognized")

        # instantiate the dataloader
        train_dataloader = DataLoader(
            data_train, batch_size=batch_size, shuffle=True)

        starter_time = time.time()

        # if trust region optimizers stores the TR optimizer as an object and instantiates the ADAM optimizer
        if isinstance(optimizer_, TRCG):
            TRCG_OP = optimizer_
            optimizer_ = torch.optim.Adam(self.model.parameters(), **kwargs)

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
                if "TRCG_OP" in locals() and epoch > optimizer.get("ADAM_epochs", -1):

                    def closure(part, total, device):

                        pred, embedding = self.model(train_batch)

                        pred = pred.to(torch.float32)
                        embedding = embedding.to(torch.float32)
                        # optimizer.zero_grad()

                        # , embedding[:, 0]).to(torch.float32)
                        loss = loss_func(train_batch, pred)
                        # loss.backward(create_graph=True)
                        # train_loss += loss.item() * pred.shape[0]
                        # total_num += pred.shape[0]

                        return loss

                    # if closure is not None:
                    loss, radius, cnt_compute, cg_iter = optimizer.step(
                        closure)
                    train_loss += loss * train_batch.shape[0]
                    total_num += train_batch.shape[0]
                    optimizer_name = "Trust Region CG"
                    # else:
                    #     optimizer.step()
                else:
                    optimizer_.zero_grad()
                    loss = loss_func(train_batch, pred)
                    loss.backward(create_graph=True)
                    train_loss += loss.item() * pred.shape[0]
                    total_num += pred.shape[0]
                    optimizer_.step()
                    optimizer_name = "Adam"

                if "verbose" in kwargs:
                    if kwargs["verbose"] == True:
                        print(f"Loss = {loss.item()}")

            train_loss /= total_num

            print(optimizer_name)
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
                                                              1, epochs, train_loss))
            print("--- %s seconds ---" % (time.time() - start_time))

            if save_all:
                torch.save(self.model.state_dict(),
                           f"{self.path}/{self.model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth")

        total_time = time.time() - starter_time

        torch.save(self.model.state_dict(),
                   f"{self.path}/{self.model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth")

        if write_CSV is not None:
            headers = ["Model Name",
                       "Optimizer",
                       "Epochs",
                       "Training_Time" "Train Loss", "Batch Size", "Loss Function", "Seed", "filename"]
            data = [self.model_name, optimizer, epochs, total_time, train_loss, batch_size, loss_func,
                    seed, f"{self.path}/{self.model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth"]
            append_to_csv(f"{self.path}/{write_CSV}", data, headers)

        self.model.eval()

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def inference_timer(self, data, batch_size=.5e4):
        torch.cuda.empty_cache()

        batch_size = int(batch_size)

        dataloader = DataLoader(data, batch_size)

        # Computes the inference time
        computeTime(self.model, dataloader, batch_size, device=self.device)

    def predict(self, data, batch_size=10000,
                single=False,
                translate_params=True):

        self.model.eval()

        dataloader = DataLoader(data, batch_size=batch_size)

        # preallocate the predictions
        num_elements = len(dataloader.dataset)
        num_batches = len(dataloader)
        data = data.clone().detach().requires_grad_(True)
        predictions = torch.zeros_like(data.clone().detach())
        params_scaled = torch.zeros((data.shape[0], 4))
        params = torch.zeros((data.shape[0], 4))

        # compute the predictions
        for i, train_batch in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size

            if i == num_batches - 1:
                end = num_elements

            pred_batch, params_scaled_, params_ = self.model(
                train_batch.to(self.device))

            predictions[start:end] = pred_batch.cpu().detach()
            params_scaled[start:end] = params_scaled_.cpu().detach()
            params[start:end] = params_.cpu().detach()

            torch.cuda.empty_cache()

        # converts negative ampltiudes to positive and shifts the phase to compensate
        if translate_params:
            params[params[:, 0] < 0, 3] = params[params[:, 0] < 0, 3] - np.pi
            params[params[:, 0] < 0, 0] = np.abs(params[params[:, 0] < 0, 0])

        if self.model.dataset.NN_phase_shift is not None:
            params_scaled[:, 3] = torch.Tensor(self.model.dataset.shift_phase(
                params_scaled[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift))
            params[:, 3] = torch.Tensor(self.model.dataset.shift_phase(
                params[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift))

        return predictions, params_scaled, params

    @staticmethod
    def mse_rankings(true, prediction, curves=False):

        def type_conversion(data):

            data = np.array(data)
            data = np.rollaxis(data, 0, data.ndim-1)

            return data

        true = type_conversion(true)
        prediction = type_conversion(prediction)

        errors = SHO_Model.MSE(prediction, true)

        index = np.argsort(errors)

        if curves:
            # true will be in the form [ranked error, channel, timestep]
            return index, errors[index], true[index], prediction[index]

        return index, errors[index]

    @staticmethod
    def MSE(true, prediction):

        # calculates the mse
        mse = np.mean((true.reshape(
            true.shape[0], -1) - prediction.reshape(true.shape[0], -1))**2, axis=1)

        # converts to a scalar if there is only one value
        if mse.shape[0] == 1:
            return mse.item()

        return mse

    @staticmethod
    def get_rankings(raw_data, pred, n=1, curves=True):
        """simple function to get the best, median and worst reconstructions

        Args:
            raw_data (np.array): array of the true values
            pred (np.array): array of the predictions
            n (int, optional): number of values for each. Defaults to 1.
            curves (bool, optional): whether to return the curves or not. Defaults to True.

        Returns:
            ind: indices of the best, median and worst reconstructions
            mse: mse of the best, median and worst reconstructions
        """
        index, mse, d1, d2 = SHO_Model.mse_rankings(
            raw_data, pred, curves=curves)
        middle_index = len(index) // 2
        start_index = middle_index - n // 2
        end_index = start_index + n

        ind = np.hstack(
            (index[:n], index[start_index:end_index], index[-n:])).flatten().astype(int)
        mse = np.hstack(
            (mse[:n], mse[start_index:end_index], mse[-n:]))

        d1 = np.stack(
            (d1[:n], d1[start_index:end_index], d1[-n:])).squeeze()
        d2 = np.stack(
            (d2[:n], d2[start_index:end_index], d2[-n:])).squeeze()

        # return ind, mse, np.swapaxes(d1[ind], 1, d1.ndim-1), np.swapaxes(d2[ind], 1, d2.ndim-1)
        return ind, mse, d1, d2

    def print_mse(self, data, labels):
        """prints the MSE of the model

        Args:
            data (tuple): tuple of datasets to calculate the MSE
            labels (list): List of strings with the names of the datasets
        """

        # loops around the dataset and labels and prints the MSE for each
        for data, label in zip(data, labels):

            if isinstance(data, torch.Tensor):
                # computes the predictions
                pred_data, scaled_param, parm = self.predict(data)
            elif isinstance(data, dict):
                pred_data, _ = self.model.dataset.get_raw_data_from_LSQF_SHO(
                    data)
                data, _ = self.model.dataset.NN_data()
                pred_data = torch.from_numpy(pred_data)

            # Computes the MSE
            out = nn.MSELoss()(data, pred_data)

            # prints the MSE
            print(f"{label} Mean Squared Error: {out:0.4f}")


@static_state_decorator
def batch_training(dataset, optimizers, noise, batch_size, epochs, seed, write_CSV="Batch_Training_Noisy_Data.csv"):

    # Generate all combinations
    combinations = list(itertools.product(
        optimizers, noise, batch_size, epochs, seed))

    for training in combinations:

        optimizer = training[0]
        noise = training[1]
        batch_size = training[2]
        epochs = training[3]
        seed = training[4]

        dataset.noise = noise

        random_seed(seed=seed)

        # constructs a test train split
        X_train, X_test, y_train, y_test = dataset.test_train_split_(
            shuffle=True)

        model_name = f"SHO_{training[0]}_noise_{training[1]}_batch_size_{training[2]}"

        print(f'Working on combination: {model_name}')

        # instantiate the model
        model = SHO_Model(dataset, training=True, model_basename=model_name)

        # fits the model
        model.fit(
            X_train,
            batch_size=batch_size,
            optimizer=optimizer,
            epochs=epochs,
            write_CSV=write_CSV,
            seed=seed,
        )


# def SHO_fit_func_nn(paramss,
#                        wvec_freq,
#                        device='cpu'):

#     Amp = paramss[:, 0].type(torch.complex128)
#     w_0 = paramss[:, 1].type(torch.complex128)
#     Q = paramss[:, 2].type(torch.complex128)
#     phi = paramss[:, 3].type(torch.complex128)
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

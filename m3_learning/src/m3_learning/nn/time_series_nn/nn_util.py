import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from IPython.display import clear_output

class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2, device="cuda"):
        """init function

        Args:
            model (PyTorch model): neural network model
            weight_decay (float): value for the weight decay
            p (int, optional): l1 regularization. Defaults to 2.
            device (str, optional): the device where the model is located. Defaults to 'cuda'.
        """

        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.device = device

    def to(self, device):
        """function that sets the device

        Args:
            device (string): PyTorch device

        Returns:
            obj: self
        """
        super().to(device)
        return self

    def forward(self, model):
        """Conducts the forward pass of the model

        Args:
            model (PyTorch model): model

        Returns:
            float: computed regularization loss
        """
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(
            self.weight_list, self.weight_decay, p=self.p
        )
        return reg_loss

    def get_weight(self, model):
        """_summary_

        Args:
            model (PyTorch model): model

        Returns:
            list: list of weights
        """
        weight_list = []
        for name, param in model.named_parameters():
            if "dec" in name and "weight" in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        """Calculates the regularization loss

        Args:
            weight_list (list): list of weights
            weight_decay (float): Sets how the regularization is decayed
            p (float): sets the norm that is used

        Returns:
            _type_: _description_
        """
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        """List of weights in layers to regularize

        Args:
            weight_list (list): list of weights
        """
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)


def loss_function(
    model,
    encoder,
    decoder,
    train_iterator,
    optimizer,
    coef=0,
    coef1=0,
    ln_parm=1,
    beta=None,
    mse=True,
    device="cuda",
):

    """Loss function

    Args:
        model (PyTorch model): model
        encoder (PyTorch model): encoder of the mode
        decoder (PyTorch model): decoder of the model
        train_iterator (iter): iterator used from straining
        optimizer (obj): optimization methods used
        coef (int, optional): used to set the lambda value of the regularization. Defaults to 0.
        coef1 (int, optional): not implemented. Defaults to 0.
        ln_parm (int, optional): norm value. Defaults to 1.
        beta (float, optional): beta for variational autoencoder. Defaults to None.
        mse (bool, optional): selects to use the MSE loss. Defaults to True.
        device (str, optional): selects the device to use. Defaults to "cuda".

    Returns:
        _type_: _description_
    """

    # regularization coefficients
    weight_decay = coef
    weight_decay_1 = coef1

    # set the training mode
    model.train()

    # loss of the epoch
    train_loss = 0
    #    for i, x in enumerate(train_iterator):
    for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

        # calculates regularization on the entire model
        reg_loss_2 = Regularization(model, weight_decay_1, p=2).to(device)

        x = x.to(device, dtype=torch.float)

        # update the gradients to zero
        optimizer.zero_grad()

        if beta is None:
            embedding = encoder(x)

        else:

            # forward pass
            #        predicted_x = model(x)
            embedding, sd, mn = encoder(x)

        if weight_decay > 0:
            reg_loss_1 = weight_decay * torch.norm(embedding, ln_parm).to(device)
        else:
            reg_loss_1 = 0.0

        predicted_x = decoder(embedding)

        if mse:
            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction="mean")
        else:
            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction="mean")

            loss = loss + reg_loss_2(model) + reg_loss_1

        # beta VAE
        if beta is not None:
            vae_loss = (
                beta * 0.5 * torch.sum(torch.exp(sd) + (mn) ** 2 - 1.0 - sd).to(device)
            )
            vae_loss /= sd.shape[0] * sd.shape[1]
        else:
            vae_loss = 0

        loss = loss + vae_loss

        # backward pass
        train_loss += loss.item()

        loss.backward()
        # update the weights
        optimizer.step()

    return train_loss


def Train(
    model,
    encoder,
    decoder,
    train_iterator,
    optimizer,
    epochs,
    coef=0,
    coef_1=0,
    ln_parm=1,
    beta=None,
    mse=True,
    device="cuda",
    save_weight = False,
):
    """Function that trains the model

    Args:
        model (Pytorch model): autoencoder model
        encoder (PyTorch model): encoder of the mode
        decoder (PyTorch model): decoder of the model
        train_iterator (iter): iterator used from straining
        optimizer (obj): optimization methods used
        epochs (int): number of epochs
        coef (int, optional): used to set the lambda value of the regularization. Defaults to 0.
        coef1 (int, optional): not implemented. Defaults to 0.
        ln_parm (int, optional): norm value. Defaults to 1.
        beta (float, optional): beta for variational autoencoder. Defaults to None.
        mse (bool, optional): selects to use the MSE loss. Defaults to True.
        device (str, optional): selects the device to use. Defaults to "cuda".

    """
    
    clear_output()
    

    N_EPOCHS = epochs
    best_train_loss = float("inf")

    for epoch in range(N_EPOCHS):

        train = loss_function(
            model,
            encoder,
            decoder,
            train_iterator,
            optimizer,
            coef,
            coef_1,
            ln_parm,
            beta,
            mse,
            device,
        )

        train_loss = train
        train_loss /= len(train_iterator)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
        print(".............................")

        if best_train_loss > train_loss:
            best_train_loss = train_loss

            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            if save_weight:
                if epoch >= 0:
                    torch.save(
                        checkpoint, f"./test__Train Loss:{train_loss:.4f}-{epoch}.pkl"
                    )


def transform_nn(data, encoder, decoder, device = 'cuda'):
    """Extracts the inference from the autoencoder

    Args:
        data (array, float): input data
        encoder (PyTorch model): encoder block
        decoder (PyTorch model): decoder block
        device (str, optional): selects the device to use. Defaults to "cuda".

    Returns:
        array, float: encoder results, decoder results
    """
    try:
        encoded_spectra = encoder(
            torch.tensor(np.atleast_3d(data), dtype=torch.float32).to(device)
        )
    except:
        pass

    try:
        encoded_spectra = encoder(torch.tensor(data, dtype=torch.float32).to(device))
    except:
        pass

    decoded_spectra = decoder(encoded_spectra)

    encoded_spectra = encoded_spectra.to("cpu")
    encoded_spectra = encoded_spectra.detach().numpy()
    decoded_spectra = decoded_spectra.to("cpu")
    decoded_spectra = decoded_spectra.detach().numpy()
    return encoded_spectra, decoded_spectra

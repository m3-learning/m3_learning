import torch


def loop_fitting_function_torch(y, V, type='9 parameters', device='cuda'):
    """Hysteresis loop fitting function using torch tensors

    Args:
        V (np.array): voltage array
        y (np.array): hysteresis loop data
        type (str, optional): loop fitting function to use. Defaults to '9 parameters'.
        device (str, optional): device to run neural network. Defaults to 'cuda'.

    Returns:
        np.array: neural network fit results
    """

    V = torch.tensor(V)

    try:
        y = torch.from_numpy(y)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, 0)
    except:
        pass

    half_len = len(V) // 2
    V = V.type(torch.float64).to(device)
    y = y.type(torch.float64).to(device)

    # print(y.shape)
    # expands the tensor
    y = y.unsqueeze(-1)#.repeat(1, 1, half_len)
    # print("y-shape")
    # print(y.shape)

    if (type == '9 parameters'):

        a0, a1, a2, a3, a4, b0, b1, b2, b3 = [y[:, i] for i in range(9)]

        V1, V2 = V[:half_len], V[half_len:]

        g1 = (b1 - b0) / 2 * (torch.erf((V1 - a2) * 1000) + 1) + b0
        g2 = (b3 - b2) / 2 * (torch.erf((V2 - a3) * 1000) + 1) + b2

        y1 = (g1 * torch.erf((V1 - a2) / g1) + b0) / (b0 + b1)
        y2 = (g2 * torch.erf((V2 - a3) / g2) + b2) / (b2 + b3)

        f1 = a0 + a1 * y1 + a4 * V1
        f2 = a0 + a1 * y2 + a4 * V2

        return torch.cat((f1, f2), axis=1)

    elif type == '13 parameters':
        Warning('13 parameters not implemented yet')
        # a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, Au, Al = [
        #     y[:, i] for i in range(13)]

        # S1 = (b1 + b2) / 2 + (b2 - b1) / 2 * torch.erf((V - b7) / b5)
        # S2 = (b4 + b3) / 2 + (b3 - b4) / 2 * torch.erf((V - b8) / b6)

        # Branch1 = (a1 + a2) / 2 + (a2 - a1) / 2 * \
        #     torch.erf((V - Au) / S1) + a3 * V
        # Branch2 = (a1 + a2) / 2 + (a2 - a1) / 2 * \
        #     torch.erf((V - Al) / S2) + a3 * V

        # return torch.squeeze(torch.cat((Branch1, torch.flipud(Branch2)), axis=0))

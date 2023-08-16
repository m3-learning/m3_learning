import torch


def loop_fitting_function_torch(V, y, type='9 parameters', device='cuda'):

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

    if(type == '9 parameters'):
        
        a0, a1, a2, a3, a4, b0, b1, b2, b3 = [y[:, i] for i in range(9)]

        V1, V2 = V[:half_len], V[half_len:]

        g1 = (b1 - b0) / 2 * (torch.erf((V1 - a2) * 1000) + 1) + b0
        g2 = (b3 - b2) / 2 * (torch.erf((V2 - a3) * 1000) + 1) + b2

        y1 = (g1 * torch.erf((V1 - a2) / g1) + b0) / (b0 + b1)
        y2 = (g2 * torch.erf((V2 - a3) / g2) + b2) / (b2 + b3)

        f1 = a0 + a1 * y1 + a4 * V1
        f2 = a0 + a1 * y2 + a4 * V2

        return torch.transpose(torch.cat((f1, f2), axis=0), 1, 0)

    elif type == '13 parameters':
        a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, Au, Al = [
            y[:, i] for i in range(13)]

        S1 = (b1 + b2) / 2 + (b2 - b1) / 2 * torch.erf((V - b7) / b5)
        S2 = (b4 + b3) / 2 + (b3 - b4) / 2 * torch.erf((V - b8) / b6)

        Branch1 = (a1 + a2) / 2 + (a2 - a1) / 2 * \
            torch.erf((V - Au) / S1) + a3 * V
        Branch2 = (a1 + a2) / 2 + (a2 - a1) / 2 * \
            torch.erf((V - Al) / S2) + a3 * V

        return torch.squeeze(torch.cat((Branch1, torch.flipud(Branch2)), axis=0))
    
        # a0 = y[:, 0].type(torch.float64)
        # a1 = y[:, 1].type(torch.float64)
        # a2 = y[:, 2].type(torch.float64)
        # a3 = y[:, 3].type(torch.float64)
        # a4 = y[:, 4].type(torch.float64)
        # b0 = y[:, 5].type(torch.float64)
        # b1 = y[:, 6].type(torch.float64)
        # b2 = y[:, 7].type(torch.float64)
        # b3 = y[:, 8].type(torch.float64)
        # d = 1000
        # V1 = torch.tensor(V[:int(len(V) / 2)]).cuda()
        # V2 = torch.tensor(V[int(len(V) / 2):]).cuda()

        # g1 = (b1 - b0) / 2 * (torch.erf((V1 - a2) * d) + 1) + b0
        # g2 = (b3 - b2) / 2 * (torch.erf((V2 - a3) * d) + 1) + b2

        # y1 = (g1 * torch.erf((V1 - a2) / g1) + b0) / (b0 + b1)
        # y2 = (g2 * torch.erf((V2 - a3) / g2) + b2) / (b2 + b3)

        # f1 = a0 + a1 * y1 + a4 * V1
        # f2 = a0 + a1 * y2 + a4 * V2

        # loop_eval = torch.transpose(torch.cat((f1, f2), axis=0), 1, 0)
        # return loop_eval
    
    elif(type == '13 parameters'):
        a1 = y[:, 0].type(torch.float64)
        a2 = y[:, 1].type(torch.float64)
        a3 = y[:, 2].type(torch.float64)
        b1 = y[:, 3].type(torch.float64)
        b2 = y[:, 4].type(torch.float64)
        b3 = y[:, 5].type(torch.float64)
        b4 = y[:, 6].type(torch.float64)
        b5 = y[:, 7].type(torch.float64)
        b6 = y[:, 8].type(torch.float64)
        b7 = y[:, 9].type(torch.float64)
        b8 = y[:, 10].type(torch.float64)
        Au = y[:, 11].type(torch.float64)
        Al = y[:, 12].type(torch.float64)

        # See supporting information for more information about the form of this function
        S1 = ((b1 + b2) / 2) + ((b2 - b1) / 2) * torch.erf((V - b7) / b5)
        S2 = ((b4 + b3) / 2) + ((b3 - b4) / 2) * torch.erf((V - b8) / b6)
        Branch1 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            torch.erf((V - Au) / S1) + a3 * V
        Branch2 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            torch.erf((V - Al) / S2) + a3 * V

        return torch.squeeze(torch.cat((Branch1, torch.flipud(Branch2)), axis=0))
    else:
        print('No such parameters')
        return None

    # if type == '9 parameters':

    #     a0, a1, a2, a3, a4, b0, b1, b2, b3 = [y[:, i] for i in range(9)]

    #     V1, V2 = V[:half_len], V[half_len:]

    #     g1 = (b1 - b0) / 2 * (torch.erf((V1 - a2) * 1000) + 1) + b0
    #     g2 = (b3 - b2) / 2 * (torch.erf((V2 - a3) * 1000) + 1) + b2

    #     y1 = (g1 * torch.erf((V1 - a2) / g1) + b0) / (b0 + b1)
    #     y2 = (g2 * torch.erf((V2 - a3) / g2) + b2) / (b2 + b3)

    #     f1 = a0 + a1 * y1 + a4 * V1
    #     f2 = a0 + a1 * y2 + a4 * V2

    #     return torch.transpose(torch.cat((f1, f2), axis=0), 1, 0)

    # elif type == '13 parameters':
    #     a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, Au, Al = [
    #         y[:, i] for i in range(13)]

    #     S1 = (b1 + b2) / 2 + (b2 - b1) / 2 * torch.erf((V - b7) / b5)
    #     S2 = (b4 + b3) / 2 + (b3 - b4) / 2 * torch.erf((V - b8) / b6)

    #     Branch1 = (a1 + a2) / 2 + (a2 - a1) / 2 * \
    #         torch.erf((V - Au) / S1) + a3 * V
    #     Branch2 = (a1 + a2) / 2 + (a2 - a1) / 2 * \
    #         torch.erf((V - Al) / S2) + a3 * V

    #     return torch.squeeze(torch.cat((Branch1, torch.flipud(Branch2)), axis=0))

    # else:
    #     print('No such parameters')
    #     return None

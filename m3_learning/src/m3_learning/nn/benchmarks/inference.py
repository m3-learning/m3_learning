import torch
import time
import numpy as np

def computeTime(model, train_dataloader, batch_size, device='cuda', write_to_file=False):
    if device == 'cuda':
        model = model.cuda()
        inputs = train_dataloader.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1

    time_print = (np.mean(time_spent)*1000)/batch_size
    print(f'Avg execution time (ms): {time_print:.6f}')

    if write_to_file:
        return f'Avg execution time (ms): {time_print:.6f}'
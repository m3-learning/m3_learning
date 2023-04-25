import torch
import time
import numpy as np


def computeTime(model, train_dataloader, batch_size, device='cuda', write_to_file=False):

    model.eval()

    time_spent = []
    for i, data in enumerate(train_dataloader, 1):
        start_time = time.time()
        with torch.no_grad():
            _ = model(data.to(device))

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)

    print(f"Mean execution time computed for {i} batches of size {batch_size}")

    averaged_time = (np.mean(time_spent)*1000)
    std_time = (np.std(time_spent)*1000)
    print(
        f'Average execution time per batch (ms): {averaged_time:.6f} ± {std_time:.6f}')
    print(
        f'Average execution time per iteration (ms): {averaged_time/batch_size:.6f} ± {std_time/batch_size:.6f}')

    print(rf'Total execution time (s): {np.sum(time_spent):.2f} ')

    if write_to_file:
        return f'Avg execution time (ms): {averaged_time:.6f}'

import torch
import numpy as np
from functools import partial
from emulator.mimo.pyct.generate_training_dataset import map2tiles, tiles2map


def rmse(y_hat, y):
    if isinstance(y_hat, torch.Tensor):
        return torch.mean(torch.sqrt((y_hat-y)**2))
    return np.mean(np.sqrt((y_hat-y)**2))


#making these 2 different functions as to not incur extra computation when computing loss for training
#Could use default threshold as with difference_map
def rmse_cloud_only(y_hat, y, threshold):
    inds = np.where((y_hat >= threshold) | (y >= threshold))

    return rmse(y_hat[inds], y[inds])
 
def tiled_rmse(y_hat, y, patch_size, equal_size_output=True, eval_func=rmse):
    y_hat_tiled = map2tiles(y_hat, patch_size, patch_size)
    y_tiled = map2tiles(y, patch_size, patch_size)

    if equal_size_output:
        output = np.zeros(y_hat_tiled.shape)
    else:
        output = np.zeros(y_hat_tiled.shape[0])

    for i in range(y_hat_tiled.shape[0]):
        mean_rmse = rmse(y_hat_tiled[i], y_tiled[i]).mean()
        output[i] = mean_rmse

    if equal_size_output:
        output = tiles2map(output, y_hat.shape[0], y_hat.shape[1], patch_size)

    return output

def tiled_rmse_cloud_only(y_hat, y, patch_size, threshold, equal_size_output=True, ):

    eval_func = partial(rmse_cloud_only, threshold=threshold)

    return tiled_rmse(y_hat, y, patch_size, equal_size_output, eval_func)



def difference_map(y_hat, y, threshold = np.inf):

    if np.isfinite(threshold):
        output = np.zeros(y_hat.shape)
        inds = np.where((y >= threshold) | (y_hat >= threshold))
        output[inds] = y_hat[inds] - y[inds]
        return output

    return y_hat - y


def tiled_difference_map(y_hat, y, patch_size, threshold=np.inf):

    y_hat_tiled = map2tiles(y_hat, patch_size, patch_size)
    y_tiled = map2tiles(y, patch_size, patch_size)
 
    output = np.zeros(y_hat_tiled.shape) 

    for i in range(y_hat_tiled.shape[0]):
        mean_diff = difference_map(y_hat_tiled[i], y_tiled[i], threshold).mean()
        output[i] = mean_diff

    output = tiles2map(output, y_hat.shape[0], y_hat.shape[1], patch_size)


    return output




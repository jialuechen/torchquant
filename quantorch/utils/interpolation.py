import torch
from torch import Tensor
from scipy.interpolate import interp1d

def linear_interpolation(x: Tensor, y: Tensor, xi: Tensor) -> Tensor:
    interp_func = interp1d(x.numpy(), y.numpy(), kind='linear')
    yi = torch.tensor(interp_func(xi.numpy()))
    return yi

def cubic_interpolation(x: Tensor, y: Tensor, xi: Tensor) -> Tensor:
    interp_func = interp1d(x.numpy(), y.numpy(), kind='cubic')
    yi = torch.tensor(interp_func(xi.numpy()))
    return yi
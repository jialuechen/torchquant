import torch
from torch import Tensor
from ..tensor import steps

def brownian_motion(size,time,inititative_value:float=0,drift:float=0,volatility:float=0)->Tensor:
    dt=torch.empty_like(time)
    return drift*time+(volatility*torch.randn(size)*dt.sqrt()).cumsum(-1)

def geometric_brownian_motion()->Tensor:
    pass
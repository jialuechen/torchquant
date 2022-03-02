from os import device_encoding
import torch
from torch import Tensor

'''
generator for brownian motion
'''
def brownian_motion(size,time,init_value:float=0.0,drift:float=0.0,volatility:float=0.0,device=None)->Tensor:
    dt=torch.empty(time)
    dwt=dt.sqrt()
    return init_value+drift*time+(volatility*torch.randn(size)*dwt).cumsum(dim=-1)

'''
generator for geometric brownian motion
'''
def geometric_brownian_motion(size,time,init_value:float=0.0,drift:float=0.0,volatility:float=0.0,device=None)->Tensor:
    return init_value*torch.exp(brownian_motion(size,time,drift-volatility**2/2,volatility,device=device))
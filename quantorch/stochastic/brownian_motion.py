from os import device_encoding
import torch
from torch import Tensor

'''
generate brownian motion
'''
def brownian_motion(size,time:Tensor,init_value:Tensor,drift:Tensor,volatility:Tensor,device=None)->Tensor:
    dt=torch.empty(time)
    dwt=dt.sqrt()
    return init_value+drift*time+(volatility*torch.randn(size)*dwt).cumsum(dim=-1)

'''
generate geometric brownian motion
'''
def geometric_brownian_motion(size,time:Tensor,init_value:Tensor,drift:Tensor,volatility:Tensor,device=None)->Tensor:
    return init_value*torch.exp(brownian_motion(size,time,drift-volatility**2/2,volatility,device=device))
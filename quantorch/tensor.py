import torch

def steps(end:float,steps=None,dtype=None,device=None)->torch.Tensor:
    return torch.linspace(0.0,end,steps+1,dtype=dtype,device=device)[1:]
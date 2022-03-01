import torch

def _parse_spot(spot:torch.Tensor=None)->torch.Tensor:
    if spot is not None:
        return spot
    

import torch
from utils.parse import _parse_spot

@torch.enable_grad
def delta(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    '''
    if kwargs.get("strike") is None and kwargs.get("spot") is None:
        kwargs["strike"]=torch.tensor(1.0)
    '''
    spot=_parse_spot(**kwargs)
    price=pricer(**kwargs)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
        )[0]

@torch.enable_grad
def gamma(pricer,create_graph:bool =False,**kwargs)->torch.tensor:
    spot=_parse_spot(**kwargs)
    the_delta=delta(pricer,create_graph=True,**kwargs).requires_grad_()
    return torch.autograd.grad(the_delta,inputs=spot,grad_outputs=torch.ones_like(the_delta),create_graph=create_graph)[0]

@torch.enable_grad
def vega(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    price=pricer(**kwargs)
    vol=kwargs["vol"].requires_grad_()
    return torch.autograd.grad(
        price,
        inputs=vol,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]
    

@torch.enable_grad
def theta(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    expiry=kwargs["expiry"].requires_grad_()
    price=pricer(**kwargs)
    return -torch.autograd.grad(
        price,
        inputs=expiry,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]

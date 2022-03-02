import torch
from utils.parse import _parse_spot

@torch.enable_grad
def delta(pricing_model,create_graph:bool=False,**kwargs)->torch.tensor:
    spot=_parse_spot(**kwargs)
    price=pricing_model(**kwargs)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
        )[0]

@torch.enable_grad
def gamma(pricing_model,create_graph:bool =False,**kwargs)->torch.tensor:
    spot=_parse_spot(**kwargs)
    the_delta=delta(pricing_model,create_graph=True,**kwargs).requires_grad_()
    return torch.autograd.grad(the_delta,inputs=spot,grad_outputs=torch.ones_like(the_delta),create_graph=create_graph)[0]

@torch.enable_grad
def vega(pricing_model,create_graph:bool=False,**kwargs)->torch.tensor:
    price=pricing_model(**kwargs)
    vol=kwargs["vol"].requires_grad_()
    return torch.autograd.grad(
        price,
        inputs=vol,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]
    

@torch.enable_grad
def theta(pricing_model,create_graph:bool=False,**kwargs)->torch.tensor:
    expiry=kwargs["expiry"].requires_grad_()
    price=pricing_model(**kwargs)
    return -torch.autograd.grad(
        price,
        inputs=expiry,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]

import torch

'''
For the kwargs below, we required the all the values in the dictionary are tensors.
Usually the basic keys in the kwargs are spot,volatility,expiry and risk-free rate. The keys should meet the requirements of the pricer.
'''

@torch.enable_grad
def delta(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    # Please note that here the spot is a tensor
    spot=kwargs["spot"].requires_grad_()
    price=pricer(**kwargs)
    return torch.autograd.grad(price,inputs=spot,grad_outputs=torch.ones_like(price),create_graph=create_graph)[0]

@torch.enable_grad
def gamma(pricer,create_graph:bool =False,**kwargs)->torch.tensor:
    spot=kwargs["spot"]
    # set create_graph as True to allow the computation of the second order derivatives
    delta_tensor=delta(pricer,create_graph=True,**kwargs).requires_grad_()
    return torch.autograd.grad(delta_tensor,inputs=spot,grad_outputs=torch.ones_like(delta_tensor),create_graph=create_graph)[0]

@torch.enable_grad
def vega(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    price=pricer(**kwargs)
    vol=kwargs["vol"].requires_grad_()
    return torch.autograd.grad(price,inputs=vol,grad_outputs=torch.ones_like(price),create_graph=create_graph)[0]
    
@torch.enable_grad
def theta(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    expiry=kwargs["expiry"].requires_grad_()
    price=pricer(**kwargs)
    return -torch.autograd.grad(price,inputs=expiry,grad_outputs=torch.ones_like(price), create_graph=create_graph)[0]

@torch.enable_grad
def rho(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    risk_free_rate=kwargs["risk_free_rate"].requires_grad_()
    price=pricer(**kwargs)
    return -torch.autograd.grad(price,inputs=risk_free_rate,grad_outputs=torch.ones_like(price), create_graph=create_graph)[0]

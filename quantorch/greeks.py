import torch

'''
For the kwargs below, we required the all the values in the dictionary are tensors.
Usually the basic keys in the kwargs are spot,volatility,maturity and rate. The keys should meet the requirements of the pricer.
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
    the_delta=delta(pricer,create_graph=True,**kwargs).requires_grad_()
    return torch.autograd.grad(the_delta,inputs=spot,grad_outputs=torch.ones_like(the_delta),create_graph=create_graph)[0]

@torch.enable_grad
def vega(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    price=pricer(**kwargs)
    vol=kwargs["vol"].requires_grad_()
    return torch.autograd.grad(price,inputs=vol,grad_outputs=torch.ones_like(price),create_graph=create_graph)[0]
    

@torch.enable_grad
def theta(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    maturity=kwargs["maturity"].requires_grad_()
    price=pricer(**kwargs)
    return -torch.autograd.grad(price,inputs=maturity,grad_outputs=torch.ones_like(price), create_graph=create_graph)[0]

@torch.enable_grad
def rho(pricer,create_graph:bool=False,**kwargs)->torch.tensor:
    rate=kwargs["rate"].requires_grad_()
    price=pricer(**kwargs)
    return -torch.autograd.grad(price,inputs=rate,grad_outputs=torch.ones_like(price), create_graph=create_graph)[0]

import torch
from torch import Tensor
from scipy.optimize import minimize

def sabr_loss(params, market_vols, strikes, expiries):
    alpha, beta, rho, nu = params
    model = SABRModel(strikes, expiries, alpha, beta, rho, nu)
    model_vols = model.price_option('call')
    loss = torch.sum((market_vols - model_vols) ** 2)
    return loss

def calibrate_sabr(market_vols: Tensor, strikes: Tensor, expiries: Tensor) -> Tensor:
    initial_params = torch.tensor([0.2, 0.5, -0.5, 0.3])
    result = minimize(sabr_loss, initial_params, args=(market_vols, strikes, expiries), method='BFGS')
    return result.x
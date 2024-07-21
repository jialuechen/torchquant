import torch
from torch import Tensor
from scipy.optimize import minimize

def heston_loss(params, market_prices, strikes, expiries, spot, rate):
    kappa, theta, sigma, rho, v0 = params
    model = HestonModel(spot, strikes, expiries, rate, kappa, theta, sigma, rho, v0)
    model_prices = model.price_option('call')
    loss = torch.sum((market_prices - model_prices) ** 2)
    return loss

def calibrate_heston(market_prices: Tensor, strikes: Tensor, expiries: Tensor, spot: Tensor, rate: Tensor) -> Tensor:
    initial_params = torch.tensor([2.0, 0.04, 0.1, -0.7, 0.04])
    result = minimize(heston_loss, initial_params, args=(market_prices, strikes, expiries, spot, rate), method='BFGS')
    return result.x
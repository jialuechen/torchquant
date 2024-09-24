import torch
from torch import Tensor
from scipy.optimize import minimize
from models.interest_rate import vasicek_model as VasicekModel

def vasicek_loss(params, market_rates, times):
    kappa, theta, sigma = params
    model = VasicekModel(market_rates[0], kappa, theta, sigma, times)
    model_rates = model.simulate()
    loss = torch.sum((market_rates - model_rates) ** 2)
    return loss

def calibrate_vasicek(market_rates: Tensor, times: Tensor) -> Tensor:
    initial_params = torch.tensor([0.5, 0.05, 0.01])
    result = minimize(vasicek_loss, initial_params, args=(market_rates, times), method='BFGS')
    return result.x
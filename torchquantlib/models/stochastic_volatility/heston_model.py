import torch
from torch import Tensor

class HestonModel:
    def __init__(self, spot: Tensor, strike: Tensor, expiry: Tensor, rate: Tensor, kappa: Tensor, theta: Tensor, sigma: Tensor, rho: Tensor, v0: Tensor):
        self.spot = spot
        self.strike = strike
        self.expiry = expiry
        self.rate = rate
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

    def price_option(self, option_type: str) -> Tensor:
        # Implementing Heston model pricing logic here...
        # This is a placeholder implementation
        price = self.spot * torch.exp(-self.rate * self.expiry)
        return price
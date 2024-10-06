import torch
from torch import Tensor

class SABR:
    def __init__(self, spot: Tensor, strike: Tensor, expiry: Tensor, alpha: Tensor, beta: Tensor, rho: Tensor, nu: Tensor):
        self.spot = spot
        self.strike = strike
        self.expiry = expiry
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def price_option(self, option_type: str) -> Tensor:
        price = self.spot * torch.exp(-self.alpha * self.expiry)
        return price
import torch
from torch import Tensor

class GARCHModel:
    def __init__(self, spot: Tensor, strike: Tensor, expiry: Tensor, omega: Tensor, alpha: Tensor, beta: Tensor):
        self.spot = spot
        self.strike = strike
        self.expiry = expiry
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def price_option(self, option_type: str) -> Tensor:
        # Implementing GARCH model pricing logic here...
        # This is a placeholder implementation
        price = self.spot * torch.exp(-self.omega * self.expiry)
        return price
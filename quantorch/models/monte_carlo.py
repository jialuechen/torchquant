import torch
from torch import Tensor

class MonteCarlo:
    def __init__(self, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, num_paths: int, num_steps: int):
        self.spot = spot
        self.strike = strike
        self.expiry = expiry
        self.volatility = volatility
        self.rate = rate
        self.num_paths = num_paths
        self.num_steps = num_steps

    def simulate(self) -> Tensor:
        dt = self.expiry / self.num_steps
        price_paths = torch.zeros((self.num_paths, self.num_steps + 1))
        price_paths[:, 0] = self.spot

        for t in range(1, self.num_steps + 1):
            z = torch.randn(self.num_paths)
            price_paths[:, t] = price_paths[:, t-1] * torch.exp((self.rate - 0.5 * self.volatility**2) * dt + self.volatility * torch.sqrt(dt) * z)

        payoff = torch.maximum(price_paths[:, -1] - self.strike, torch.tensor(0.0))
        price = torch.exp(-self.rate * self.expiry) * payoff.mean()
        return price
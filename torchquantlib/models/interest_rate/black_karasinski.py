import torch
from torch import Tensor

class BlackKarasinskiModel:
    def __init__(self, spot_rate: Tensor, kappa: Tensor, theta: Tensor, sigma: Tensor, time: Tensor):
        self.spot_rate = spot_rate
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.time = time

    def simulate(self) -> Tensor:
        dt = 1 / self.time
        rates = torch.zeros(self.time)
        rates[0] = self.spot_rate

        for t in range(1, self.time):
            rates[t] = rates[t-1] + self.kappa * (self.theta - rates[t-1]) * dt + self.sigma * torch.sqrt(dt) * torch.randn(1)
        
        return rates
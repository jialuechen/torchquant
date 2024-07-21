import torch
from torch import Tensor

class LMMModel:
    def __init__(self, forward_rates: Tensor, volatilities: Tensor, times: Tensor):
        self.forward_rates = forward_rates
        self.volatilities = volatilities
        self.times = times

    def simulate(self) -> Tensor:
        dt = 1 / self.times
        rates = torch.zeros(len(self.forward_rates), self.times)
        rates[:, 0] = self.forward_rates

        for t in range(1, self.times):
            for i in range(len(self.forward_rates)):
                rates[i, t] = rates[i, t-1] + self.volatilities[i] * torch.sqrt(dt) * torch.randn(1)
        
        return rates
import torch
from models.stochastic_model import StochasticModel

class DupireLocalVol(StochasticModel):
    """
    Dupire local volatility model.
    """
    def __init__(self, local_vol_func):
        # local_vol_func: a function that takes (S, t) and returns local volatility σ(S, t)
        self.local_vol_func = local_vol_func
        super().__init__(params={})

    def simulate(self, S0, T, N, steps=100):
        dt = T / steps
        dt = torch.tensor(dt, device=self.device)
        S0 = torch.tensor(S0, device=self.device)
        N = int(N)
        steps = int(steps)

        S = torch.zeros(N, steps, device=self.device)
        S[:, 0] = S0

        for t in range(1, steps):
            t_curr = t * dt
            S_t_minus = S[:, t - 1]
            sigma_t = self.local_vol_func(S_t_minus, t_curr)
            Z = torch.randn(N, device=self.device)
            dS = sigma_t * S_t_minus * torch.sqrt(dt) * Z
            S[:, t] = S_t_minus + dS

        return S[:, -1]
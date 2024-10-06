import torch
from models.stochastic_model import StochasticModel

class BlackKarasinski(StochasticModel):
    """
    Black-Karasinski interest rate model.
    """
    def __init__(self, a_init=0.1, sigma_init=0.01, r0_init=0.03):
        params = {
            'a': torch.tensor(a_init, requires_grad=True),
            'sigma': torch.tensor(sigma_init, requires_grad=True),
            'r0': torch.tensor(r0_init, requires_grad=True)
        }
        super().__init__(params)

    def simulate(self, S0, T, N, steps=100):
        dt = T / steps
        a = self.params['a']
        sigma = self.params['sigma']
        r0 = self.params['r0']

        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)

        a = torch.clamp(a, min=1e-6)
        sigma = torch.clamp(sigma, min=1e-6)

        log_r = torch.zeros(N, steps, device=self.device)
        log_r[:, 0] = torch.log(r0)

        for t in range(1, steps):
            log_r_t_minus = log_r[:, t - 1]
            dlog_r = -a * log_r_t_minus * dt + sigma * torch.sqrt(dt) * torch.randn(N, device=self.device)
            log_r[:, t] = log_r_t_minus + dlog_r

        r = torch.exp(log_r)
        return r[:, -1]

    def _apply_constraints(self):
        self.params['a'].data.clamp_(min=1e-6)
        self.params['sigma'].data.clamp_(min=1e-6)

import torch
from models.stochastic_model import StochasticModel

class CIR(StochasticModel):
    """
    Cox-Ingersoll-Ross (CIR) interest rate model.
    """
    def __init__(self, kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03):
        params = {
            'kappa': torch.tensor(kappa_init, requires_grad=True),
            'theta': torch.tensor(theta_init, requires_grad=True),
            'sigma': torch.tensor(sigma_init, requires_grad=True),
            'r0': torch.tensor(r0_init, requires_grad=True)
        }
        super().__init__(params)

    def simulate(self, S0, T, N, steps=100):
        dt = T / steps
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        r0 = self.params['r0']

        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)

        kappa = torch.clamp(kappa, min=1e-6)
        theta = torch.clamp(theta, min=1e-6)
        sigma = torch.clamp(sigma, min=1e-6)

        r = torch.zeros(N, steps, device=self.device)
        r[:, 0] = r0

        for t in range(1, steps):
            r_t_minus = torch.relu(r[:, t - 1])  # Ensure positivity
            dr = kappa * (theta - r_t_minus) * dt + sigma * torch.sqrt(r_t_minus * dt) * torch.randn(N, device=self.device)
            r[:, t] = r_t_minus + dr

        return r[:, -1]

    def _apply_constraints(self):
        self.params['kappa'].data.clamp_(min=1e-6)
        self.params['theta'].data.clamp_(min=1e-6)
        self.params['sigma'].data.clamp_(min=1e-6)
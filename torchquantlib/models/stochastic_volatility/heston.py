import torch
from models.stochastic_model import StochasticModel
class Heston(StochasticModel):
    """
    Heston stochastic volatility model.
    """
    def __init__(self, kappa_init=1.0, theta_init=0.04, sigma_v_init=0.5, rho_init=-0.5, v0_init=0.04, mu_init=0.0):
        params = {
            'kappa': torch.tensor(kappa_init, requires_grad=True),
            'theta': torch.tensor(theta_init, requires_grad=True),
            'sigma_v': torch.tensor(sigma_v_init, requires_grad=True),
            'rho': torch.tensor(rho_init, requires_grad=True),
            'v0': torch.tensor(v0_init, requires_grad=True),
            'mu': torch.tensor(mu_init, requires_grad=True)
        }
        super().__init__(params)

    def simulate(self, S0, T, N, steps=100):
        dt = T / steps
        mu = self.params['mu']
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma_v = self.params['sigma_v']
        rho = torch.clamp(self.params['rho'], min=-0.999, max=0.999)
        v0 = self.params['v0']

        S0 = torch.tensor(S0, device=self.device)
        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)

        # Ensure parameters are positive and move to device
        theta = torch.clamp(theta, min=1e-6)
        sigma_v = torch.clamp(sigma_v, min=1e-6)
        v0 = torch.clamp(v0, min=1e-6)

        S = torch.zeros(N, steps, device=self.device)
        v = torch.zeros(N, steps, device=self.device)
        S[:, 0] = S0
        v[:, 0] = v0

        for t in range(1, steps):
            Z1 = torch.randn(N, device=self.device)
            Z2 = torch.randn(N, device=self.device)
            W_S = Z1 * torch.sqrt(dt)
            W_v = (rho * Z1 + torch.sqrt(1 - rho ** 2) * Z2) * torch.sqrt(dt)

            v_t_minus = torch.relu(v[:, t - 1])

            dv = kappa * (theta - v_t_minus) * dt + sigma_v * torch.sqrt(v_t_minus) * W_v
            v[:, t] = v_t_minus + dv

            dS = S[:, t - 1] * (mu * dt + torch.sqrt(v_t_minus) * W_S)
            S[:, t] = S[:, t - 1] + dS

        return S[:, -1]

    def _apply_constraints(self):
        self.params['theta'].data.clamp_(min=1e-6)
        self.params['sigma_v'].data.clamp_(min=1e-6)
        self.params['v0'].data.clamp_(min=1e-6)
        self.params['rho'].data.clamp_(min=-0.999, max=0.999)


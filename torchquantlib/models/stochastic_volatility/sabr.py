import torch
from models.stochastic_model import StochasticModel

class SABR(StochasticModel):
    """
    SABR model.
    """
    def __init__(self, alpha_init=0.2, beta_init=0.5, rho_init=0.0, nu_init=0.3, F0=100.0):
        params = {
            'alpha': torch.tensor(alpha_init, requires_grad=True),
            'beta': torch.tensor(beta_init, requires_grad=True),
            'rho': torch.tensor(rho_init, requires_grad=True),
            'nu': torch.tensor(nu_init, requires_grad=True),
        }
        super().__init__(params)
        self.F0 = torch.tensor(F0, device=self.device)

    def simulate(self, S0, T, N, steps=100):
        dt = T / steps
        alpha = self.params['alpha']
        beta = torch.clamp(self.params['beta'], min=0.0, max=1.0)
        rho = torch.clamp(self.params['rho'], min=-0.999, max=0.999)
        nu = self.params['nu']

        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)

        F = torch.zeros(N, steps, device=self.device)
        alpha_t = torch.zeros(N, steps, device=self.device)
        F[:, 0] = self.F0
        alpha_t[:, 0] = alpha

        for t in range(1, steps):
            Z1 = torch.randn(N, device=self.device)
            Z2 = torch.randn(N, device=self.device)
            W_F = Z1 * torch.sqrt(dt)
            W_alpha = (rho * Z1 + torch.sqrt(1 - rho ** 2) * Z2) * torch.sqrt(dt)

            F_t_minus = F[:, t - 1]
            alpha_t_minus = alpha_t[:, t - 1]

            dF = alpha_t_minus * (F_t_minus ** beta) * W_F
            F[:, t] = F_t_minus + dF

            dalpha = nu * alpha_t_minus * W_alpha
            alpha_t[:, t] = alpha_t_minus + dalpha

        return F[:, -1]

    def option_price(self, K, T, r, option_type='call', N=10000, steps=100):
        """
        Price a European option using Monte Carlo simulation under the SABR model.

        Parameters:
        - K: Strike price
        - T: Time to maturity
        - r: Risk-free interest rate
        - option_type: 'call' or 'put'
        - N: Number of simulation paths
        - steps: Number of time steps in simulation

        Returns:
        - Option price
        """
        F_T = self.simulate(S0=self.F0, T=T, N=N, steps=steps)
        if option_type.lower() == 'call':
            payoff = torch.relu(F_T - K)
        elif option_type.lower() == 'put':
            payoff = torch.relu(K - F_T)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        price = torch.exp(-r * T) * torch.mean(payoff)
        return price.item()

    def _apply_constraints(self):
        self.params['alpha'].data.clamp_(min=1e-6)
        self.params['beta'].data.clamp_(min=0.0, max=1.0)
        self.params['nu'].data.clamp_(min=1e-6)
        self.params['rho'].data.clamp_(min=-0.999, max=0.999)
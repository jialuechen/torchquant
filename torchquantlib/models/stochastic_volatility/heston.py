import torch
from models.stochastic_model import StochasticModel

class Heston(StochasticModel):
    """
    Heston stochastic volatility model.

    This model describes the evolution of an asset price and its variance using two coupled stochastic differential equations:
    dS = μSdt + √vSdW_S
    dv = κ(θ - v)dt + σ_v√vdW_v

    where:
    S is the asset price
    v is the variance
    μ is the drift of the asset price
    κ is the rate of mean reversion of variance
    θ is the long-term mean of variance
    σ_v is the volatility of variance
    ρ is the correlation between the two Wiener processes W_S and W_v
    """

    def __init__(self, kappa_init=1.0, theta_init=0.04, sigma_v_init=0.5, rho_init=-0.5, v0_init=0.04, mu_init=0.0):
        """
        Initialize the Heston model.

        Args:
            kappa_init (float): Initial value for rate of mean reversion of variance.
            theta_init (float): Initial value for long-term mean of variance.
            sigma_v_init (float): Initial value for volatility of variance.
            rho_init (float): Initial value for correlation between asset and variance processes.
            v0_init (float): Initial value for variance.
            mu_init (float): Initial value for drift of asset price.
        """
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
        """
        Simulate asset price paths using the Heston model.

        Args:
            S0 (float): Initial asset price.
            T (float): Time horizon for simulation.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in each path.

        Returns:
            torch.Tensor: Simulated asset prices at time T.
        """
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

        # Ensure parameters are positive
        theta = torch.clamp(theta, min=1e-6)
        sigma_v = torch.clamp(sigma_v, min=1e-6)
        v0 = torch.clamp(v0, min=1e-6)

        # Initialize asset price and variance paths
        S = torch.zeros(N, steps, device=self.device)
        v = torch.zeros(N, steps, device=self.device)
        S[:, 0] = S0
        v[:, 0] = v0

        for t in range(1, steps):
            # Generate correlated random numbers
            Z1 = torch.randn(N, device=self.device)
            Z2 = torch.randn(N, device=self.device)
            W_S = Z1 * torch.sqrt(dt)
            W_v = (rho * Z1 + torch.sqrt(1 - rho ** 2) * Z2) * torch.sqrt(dt)

            v_t_minus = torch.relu(v[:, t - 1])  # Ensure non-negative variance

            # Update variance
            dv = kappa * (theta - v_t_minus) * dt + sigma_v * torch.sqrt(v_t_minus) * W_v
            v[:, t] = v_t_minus + dv

            # Update asset price
            dS = S[:, t - 1] * (mu * dt + torch.sqrt(v_t_minus) * W_S)
            S[:, t] = S[:, t - 1] + dS

        return S[:, -1]

    def option_price(self, S0, K, T, r, option_type='call', N=10000, steps=100):
        """
        Price a European option using Monte Carlo simulation under the Heston model.

        Args:
            S0 (float): Initial asset price.
            K (float): Strike price.
            T (float): Time to maturity.
            r (float): Risk-free interest rate.
            option_type (str): 'call' or 'put'.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in simulation.

        Returns:
            float: Option price.
        """
        S_T = self.simulate(S0, T, N, steps)
        if option_type.lower() == 'call':
            payoff = torch.relu(S_T - K)
        elif option_type.lower() == 'put':
            payoff = torch.relu(K - S_T)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        price = torch.exp(-r * T) * torch.mean(payoff)
        return price.item()

    def _apply_constraints(self):
        """
        Apply constraints to model parameters to ensure they remain in valid ranges.
        """
        self.params['theta'].data.clamp_(min=1e-6)
        self.params['sigma_v'].data.clamp_(min=1e-6)
        self.params['v0'].data.clamp_(min=1e-6)
        self.params['rho'].data.clamp_(min=-0.999, max=0.999)
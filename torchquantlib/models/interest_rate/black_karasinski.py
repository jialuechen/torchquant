import torch
from models.stochastic_model import StochasticModel

class BlackKarasinski(StochasticModel):
    """
    Black-Karasinski interest rate model.

    This model describes the evolution of interest rates using the following stochastic differential equation:
    d(ln(r)) = (θ(t) - a * ln(r)) * dt + σ * dW

    where:
    r is the short rate
    θ(t) is a function chosen to fit the initial term structure
    a is the mean reversion speed
    σ is the volatility
    W is a Wiener process
    """

    def __init__(self, a_init=0.1, sigma_init=0.01, r0_init=0.03):
        """
        Initialize the Black-Karasinski model.

        Args:
            a_init (float): Initial value for mean reversion speed.
            sigma_init (float): Initial value for volatility.
            r0_init (float): Initial value for short rate.
        """
        params = {
            'a': torch.tensor(a_init, requires_grad=True),
            'sigma': torch.tensor(sigma_init, requires_grad=True),
            'r0': torch.tensor(r0_init, requires_grad=True)
        }
        super().__init__(params)

    def simulate(self, S0, T, N, steps=100):
        """
        Simulate interest rate paths using the Black-Karasinski model.

        Args:
            S0 (float): Initial asset price (not used in this model, included for consistency).
            T (float): Time horizon for simulation.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in each path.

        Returns:
            torch.Tensor: Simulated interest rates at time T.
        """
        dt = T / steps
        a = self.params['a']
        sigma = self.params['sigma']
        r0 = self.params['r0']

        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)

        # Ensure positive values for mean reversion and volatility
        a = torch.clamp(a, min=1e-6)
        sigma = torch.clamp(sigma, min=1e-6)

        # Initialize log interest rates
        log_r = torch.zeros(N, steps, device=self.device)
        log_r[:, 0] = torch.log(r0)

        # Simulate log interest rate paths
        for t in range(1, steps):
            log_r_t_minus = log_r[:, t - 1]
            dlog_r = -a * log_r_t_minus * dt + sigma * torch.sqrt(dt) * torch.randn(N, device=self.device)
            log_r[:, t] = log_r_t_minus + dlog_r

        # Convert log rates back to rates
        r = torch.exp(log_r)
        return r[:, -1]

    def _apply_constraints(self):
        """
        Apply constraints to model parameters to ensure they remain in valid ranges.
        """
        self.params['a'].data.clamp_(min=1e-6)
        self.params['sigma'].data.clamp_(min=1e-6)
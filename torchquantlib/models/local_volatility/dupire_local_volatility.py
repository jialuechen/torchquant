import torch
from models.stochastic_model import StochasticModel

class DupireLocalVol(StochasticModel):
    """
    Dupire local volatility model.

    This model extends the Black-Scholes model by allowing the volatility to be a function
    of both the underlying asset price and time. It is based on Dupire's formula, which
    relates the local volatility to the implied volatility surface.

    The model is described by the following stochastic differential equation:
    dS = μSdt + σ(S,t)SdW

    where:
    S is the asset price
    μ is the drift (usually the risk-free rate)
    σ(S,t) is the local volatility function
    W is a Wiener process
    """

    def __init__(self, local_vol_func):
        """
        Initialize the Dupire Local Volatility model.

        Args:
            local_vol_func (callable): A function that takes (S, t) and returns local volatility σ(S, t).
                                       S can be a tensor of asset prices, and t is a scalar time value.
        """
        self.local_vol_func = local_vol_func
        super().__init__(params={})

    def simulate(self, S0, T, N, steps=100):
        """
        Simulate asset price paths using the Dupire Local Volatility model.

        Args:
            S0 (float): Initial asset price.
            T (float): Time horizon for simulation.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in each path.

        Returns:
            torch.Tensor: Simulated asset prices at time T.
        """
        dt = T / steps
        dt = torch.tensor(dt, device=self.device)
        S0 = torch.tensor(S0, device=self.device)
        N = int(N)
        steps = int(steps)

        # Initialize asset price paths
        S = torch.zeros(N, steps, device=self.device)
        S[:, 0] = S0

        # Simulate asset price paths
        for t in range(1, steps):
            t_curr = t * dt
            S_t_minus = S[:, t - 1]
            
            # Calculate local volatility for current asset prices and time
            sigma_t = self.local_vol_func(S_t_minus, t_curr)
            
            # Generate random normal variables for the Wiener process
            Z = torch.randn(N, device=self.device)
            
            # Calculate price increment (note: drift term is omitted for simplicity)
            dS = sigma_t * S_t_minus * torch.sqrt(dt) * Z
            
            # Update asset prices
            S[:, t] = S_t_minus + dS

        return S[:, -1]
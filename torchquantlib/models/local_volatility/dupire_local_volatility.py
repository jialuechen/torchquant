import torch
from torch import Tensor

class StochasticModel:
    def __init__(self, params):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    Limitations:
        - Uses Euler discretization, which can be inaccurate for larger time steps or higher volatility.
        - Assumes a continuous-time model.
    """

    def __init__(self, local_vol_func):
        """
        Initialize the Dupire Local Volatility model.

        Args:
            local_vol_func (callable): A function that takes (S, t, device) and returns local volatility σ(S, t).
                                       S can be a tensor of asset prices, and t is a scalar time value.
                                       The function should also handle the device.
        """
        self.local_vol_func = local_vol_func
        super().__init__(params={})

    def simulate(self, S0, T, N, rate, steps=100):
        """
        Simulate asset price paths using the Dupire Local Volatility model.

        Args:
            S0 (float): Initial asset price.
            T (float): Time horizon for simulation.
            N (int): Number of simulation paths.
            rate (float): Risk-free interest rate.
            steps (int): Number of time steps in each path.

        Returns:
            torch.Tensor: Simulated asset prices at time T.
        """
        if not isinstance(S0, (int, float)):
            raise TypeError("S0 must be a number.")
        if not isinstance(T, (int, float)):
            raise TypeError("T must be a number.")
        if not isinstance(N, int):
            raise TypeError("N must be an integer.")
        if not isinstance(rate, (int, float)):
            raise TypeError("rate must be a number.")
        dt = T / steps
        dt = torch.tensor(dt, device=self.device)
        S0 = torch.tensor(S0, device=self.device)
        rate = torch.tensor(rate, device=self.device)
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
            sigma_t = self.local_vol_func(S_t_minus, t_curr, self.device)
            
            # Generate random normal variables for the Wiener process
            Z = torch.randn(N, device=self.device)
            
            # Calculate price increment
            dS = rate * S_t_minus * dt + sigma_t * S_t_minus * torch.sqrt(dt) * Z
            
            # Update asset prices
            S[:, t] = S_t_minus + dS

        return S[:, -1]

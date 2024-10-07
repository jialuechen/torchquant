import torch
from models.stochastic_model import StochasticModel

class SABR(StochasticModel):
    """
    SABR (Stochastic Alpha, Beta, Rho) model.

    This model describes the evolution of a forward rate F and its volatility α using two coupled stochastic differential equations:
    dF = α * F^β * dW_1
    dα = ν * α * dW_2

    where:
    F is the forward rate
    α is the stochastic volatility
    β is the elasticity parameter (0 ≤ β ≤ 1)
    ν is the volatility of volatility
    ρ is the correlation between W_1 and W_2 (the two Wiener processes)
    """

    def __init__(self, alpha_init=0.2, beta_init=0.5, rho_init=0.0, nu_init=0.3, F0=100.0):
        """
        Initialize the SABR model.

        Args:
            alpha_init (float): Initial value for volatility.
            beta_init (float): Initial value for elasticity parameter (0 ≤ β ≤ 1).
            rho_init (float): Initial value for correlation between F and α processes.
            nu_init (float): Initial value for volatility of volatility.
            F0 (float): Initial forward rate.
        """
        params = {
            'alpha': torch.tensor(alpha_init, requires_grad=True),
            'beta': torch.tensor(beta_init, requires_grad=True),
            'rho': torch.tensor(rho_init, requires_grad=True),
            'nu': torch.tensor(nu_init, requires_grad=True),
        }
        super().__init__(params)
        self.F0 = torch.tensor(F0, device=self.device)

    def simulate(self, S0, T, N, steps=100):
        """
        Simulate forward rate paths using the SABR model.

        Args:
            S0 (float): Initial asset price (not used, included for consistency with other models).
            T (float): Time horizon for simulation.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in each path.

        Returns:
            torch.Tensor: Simulated forward rates at time T.
        """
        dt = T / steps
        alpha = self.params['alpha']
        beta = torch.clamp(self.params['beta'], min=0.0, max=1.0)
        rho = torch.clamp(self.params['rho'], min=-0.999, max=0.999)
        nu = self.params['nu']

        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)

        # Initialize forward rate and volatility paths
        F = torch.zeros(N, steps, device=self.device)
        alpha_t = torch.zeros(N, steps, device=self.device)
        F[:, 0] = self.F0
        alpha_t[:, 0] = alpha

        for t in range(1, steps):
            # Generate correlated random numbers
            Z1 = torch.randn(N, device=self.device)
            Z2 = torch.randn(N, device=self.device)
            W_F = Z1 * torch.sqrt(dt)
            W_alpha = (rho * Z1 + torch.sqrt(1 - rho ** 2) * Z2) * torch.sqrt(dt)

            F_t_minus = F[:, t - 1]
            alpha_t_minus = alpha_t[:, t - 1]

            # Update forward rate
            dF = alpha_t_minus * (F_t_minus ** beta) * W_F
            F[:, t] = F_t_minus + dF

            # Update volatility
            dalpha = nu * alpha_t_minus * W_alpha
            alpha_t[:, t] = alpha_t_minus + dalpha

        return F[:, -1]

    def option_price(self, K, T, r, option_type='call', N=10000, steps=100):
        """
        Price a European option using Monte Carlo simulation under the SABR model.

        Args:
            K (float): Strike price.
            T (float): Time to maturity.
            r (float): Risk-free interest rate.
            option_type (str): 'call' or 'put'.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in simulation.

        Returns:
            float: Option price.
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
        """
        Apply constraints to model parameters to ensure they remain in valid ranges.
        """
        self.params['alpha'].data.clamp_(min=1e-6)
        self.params['beta'].data.clamp_(min=0.0, max=1.0)
        self.params['nu'].data.clamp_(min=1e-6)
        self.params['rho'].data.clamp_(min=-0.999, max=0.999)
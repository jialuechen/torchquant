import torch
from models.stochastic_model import StochasticModel

class LMM(StochasticModel):
    """
    Libor Market Model (LMM).

    This model simulates the evolution of forward LIBOR rates. It's a multi-factor model
    that captures the dynamics of the entire yield curve.

    The LMM is described by the following stochastic differential equation:
    dF_i(t) = μ_i(t, F(t))dt + σ_i(t)F_i(t)dW_i(t)

    where:
    F_i(t) is the forward rate for the i-th period
    μ_i(t, F(t)) is the drift term
    σ_i(t) is the volatility of the i-th forward rate
    W_i(t) are correlated Brownian motions
    """

    def __init__(self, forward_rates_init, volatilities_init, correlations_init):
        """
        Initialize the Libor Market Model.

        Args:
            forward_rates_init (list or np.array): Initial forward rates.
            volatilities_init (list or np.array): Initial volatilities for each forward rate.
            correlations_init (np.array): Correlation matrix for the forward rates.
        """
        self.num_rates = len(forward_rates_init)
        self.forward_rates = torch.tensor(forward_rates_init, requires_grad=True, device=self.device)
        self.volatilities = torch.tensor(volatilities_init, requires_grad=True, device=self.device)
        self.correlation_matrix = torch.tensor(correlations_init, requires_grad=False, device=self.device)
        params = {
            'forward_rates': self.forward_rates,
            'volatilities': self.volatilities,
        }
        super().__init__(params)

    def simulate(self, S0, T, N, steps=100):
        """
        Simulate forward rate paths using the Libor Market Model.

        Args:
            S0 (float): Initial asset price (not used in this model, included for consistency).
            T (float): Time horizon for simulation.
            N (int): Number of simulation paths.
            steps (int): Number of time steps in each path.

        Returns:
            torch.Tensor: Simulated forward rates at time T for all tenors.
        """
        dt = T / steps
        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)
        num_rates = self.num_rates

        forward_rates = self.forward_rates
        volatilities = self.volatilities
        corr_matrix = self.correlation_matrix

        # Cholesky decomposition of the correlation matrix for generating correlated random numbers
        chol_matrix = torch.linalg.cholesky(corr_matrix)

        # Initialize forward rates tensor
        F = torch.zeros(N, num_rates, steps, device=self.device)
        F[:, :, 0] = forward_rates

        for t in range(1, steps):
            dtau = dt  # Assuming tenor intervals equal to dt
            Z = torch.randn(N, num_rates, device=self.device)
            W = Z @ chol_matrix.T  # Generate correlated Brownian increments

            for i in range(num_rates):
                F_t_minus = F[:, i, t - 1]
                sigma_i = volatilities[i]
                
                # Calculate drift term
                drift = torch.zeros(N, device=self.device)
                for j in range(i + 1, num_rates):
                    sigma_j = volatilities[j]
                    corr_ij = corr_matrix[i, j]
                    F_j = F[:, j, t - 1]
                    drift += dtau * sigma_i * sigma_j * corr_ij * F_j / (1 + dtau * F_j)

                # Update forward rates
                dF = drift * dt - sigma_i * F_t_minus * W[:, i] * torch.sqrt(dt)
                F[:, i, t] = F_t_minus + dF

        return F[:, :, -1]

    def _apply_constraints(self):
        """
        Apply constraints to model parameters to ensure they remain in valid ranges.
        """
        # Ensure volatilities remain positive
        self.volatilities.data.clamp_(min=1e-6)
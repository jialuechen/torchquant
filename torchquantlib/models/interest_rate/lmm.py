import torch
from models.stochastic_model import StochasticModel

class LMM(StochasticModel):
    """
    Libor Market Model (LMM).
    """
    def __init__(self, forward_rates_init, volatilities_init, correlations_init):
        """
        forward_rates_init: Initial forward rates (list or numpy array)
        volatilities_init: Initial volatilities for each forward rate (list or numpy array)
        correlations_init: Correlation matrix (numpy array)
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
        dt = T / steps
        dt = torch.tensor(dt, device=self.device)
        N = int(N)
        steps = int(steps)
        num_rates = self.num_rates

        forward_rates = self.forward_rates
        volatilities = self.volatilities
        corr_matrix = self.correlation_matrix

        # Cholesky decomposition of the correlation matrix
        chol_matrix = torch.linalg.cholesky(corr_matrix)

        F = torch.zeros(N, num_rates, steps, device=self.device)
        F[:, :, 0] = forward_rates

        for t in range(1, steps):
            dtau = dt  # Assuming tenor intervals equal to dt
            Z = torch.randn(N, num_rates, device=self.device)
            W = Z @ chol_matrix.T  # Correlated Brownian increments

            for i in range(num_rates):
                F_t_minus = F[:, i, t - 1]
                sigma_i = volatilities[i]
                drift = torch.zeros(N, device=self.device)
                for j in range(i + 1, num_rates):
                    sigma_j = volatilities[j]
                    corr_ij = corr_matrix[i, j]
                    F_j = F[:, j, t - 1]
                    drift += dtau * sigma_i * sigma_j * corr_ij * F_j / (1 + dtau * F_j)

                dF = drift * dt - sigma_i * F_t_minus * W[:, i] * torch.sqrt(dt)
                F[:, i, t] = F_t_minus + dF

        return F[:, :, -1]

    def _apply_constraints(self):
        # Ensure volatilities remain positive
        self.volatilities.data.clamp_(min=1e-6)
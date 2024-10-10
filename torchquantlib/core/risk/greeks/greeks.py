import torch

class MalliavinGreeks:
    """
    A class for calculating option Greeks using Malliavin calculus.
    This approach is particularly useful for complex options where closed-form solutions are not available.
    """

    def __init__(self, device=None):
        """
        Initialize the MalliavinGreeks class.

        Args:
            device (torch.device, optional): The device to run calculations on. Defaults to GPU if available, else CPU.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def european_option_greeks(self, S0, K, T, r, sigma, num_paths=100000, seed=42):
        """
        Calculate Greeks for European options using Malliavin calculus.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            num_paths (int): Number of Monte Carlo paths
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, Vega, Theta, and Rho
        """
        torch.manual_seed(seed)

        dt = T
        z = torch.randn(num_paths, device=self.device)
        
        # Simulate stock price at maturity
        ST = S0 * torch.exp((r - 0.5 * sigma**2) * T + sigma * torch.sqrt(dt) * z)
        
        # Calculate option payoff
        payoff = torch.relu(ST - K)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = ST * z / torch.sqrt(dt)
        D2t_ST = Dt_ST * (z / torch.sqrt(dt) - sigma * torch.sqrt(dt))
        D_sigma_ST = ST * (z * torch.sqrt(dt) - sigma * dt)
        D_r_ST = ST * T

        # Calculate Greeks
        Delta = discount_factor * torch.mean(payoff * Dt_ST / (sigma * S0))
        Gamma = discount_factor * torch.mean(payoff * D2t_ST / (sigma**2 * S0**2))
        Vega = discount_factor * torch.mean(payoff * D_sigma_ST / sigma)
        Theta = -discount_factor * torch.mean(payoff * ((r - 0.5 * sigma**2) * ST + 0.5 * sigma * Dt_ST)) / T
        Rho = discount_factor * torch.mean(payoff * D_r_ST) - T * discount_factor * torch.mean(payoff)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item(),
            'Theta': Theta.item(),
            'Rho': Rho.item()
        }

    def digital_option_greeks(self, S0, K, T, r, sigma, Q=10.0, num_paths=100000, epsilon=1e-4, seed=42):
        """
        Calculate Greeks for Digital options using Malliavin calculus.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            Q (float): Payout amount
            num_paths (int): Number of Monte Carlo paths
            epsilon (float): Small value for approximating Dirac delta function
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega
        """
        torch.manual_seed(seed)

        dt = T
        z = torch.randn(num_paths, device=self.device)
        
        # Simulate stock price at maturity
        ST = S0 * torch.exp((r - 0.5 * sigma**2) * T + sigma * torch.sqrt(dt) * z)

        # Approximate Dirac delta function
        delta_approx = (1 / (epsilon * torch.sqrt(2 * torch.pi))) * torch.exp(-0.5 * ((ST - K) / epsilon)**2)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = ST * z / torch.sqrt(dt)
        D2t_ST = Dt_ST * (z / torch.sqrt(dt) - sigma * torch.sqrt(dt))
        D_sigma_ST = ST * (z * torch.sqrt(dt) - sigma * dt)

        # Calculate Greeks
        Delta = discount_factor * torch.mean(Q * delta_approx * Dt_ST / (sigma * S0))
        Gamma = discount_factor * torch.mean(Q * delta_approx * D2t_ST / (sigma**2 * S0**2))
        Vega = discount_factor * torch.mean(Q * delta_approx * D_sigma_ST / sigma)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item()
        }

    def barrier_option_greeks(self, S0, K, H, T, r, sigma, option_type='up-and-out', num_paths=100000, num_steps=50, seed=42):
        """
        Calculate Greeks for Barrier options using Malliavin calculus.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            H (float): Barrier level
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): Type of barrier option ('up-and-out' or 'down-and-out')
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps in the simulation
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega
        """
        torch.manual_seed(seed)

        dt = T / num_steps
        S = torch.zeros((num_paths, num_steps+1), device=self.device)
        S[:, 0] = S0
        z = torch.randn((num_paths, num_steps), device=self.device)

        # Simulate stock price paths
        for i in range(num_steps):
            S[:, i+1] = S[:, i] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z[:, i])

        # Check if barrier is hit
        if option_type == 'up-and-out':
            hit_barrier = torch.any(S[:, 1:] >= H, dim=1)
        elif option_type == 'down-and-out':
            hit_barrier = torch.any(S[:, 1:] <= H, dim=1)
        else:
            raise ValueError("Invalid option_type. Use 'up-and-out' or 'down-and-out'.")

        # Calculate payoff
        payoff = torch.relu(S[:, -1] - K)
        payoff = payoff * (~hit_barrier)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = S[:, -1] * torch.sum(z, dim=1) / torch.sqrt(dt)
        Dt_ST = Dt_ST * (~hit_barrier)
        D2t_ST = Dt_ST * (torch.sum(z, dim=1) / torch.sqrt(dt) - sigma * torch.sqrt(dt))
        D2t_ST = D2t_ST * (~hit_barrier)
        D_sigma_ST = S[:, -1] * (torch.sum(z**2 - 1, dim=1) * torch.sqrt(dt))
        D_sigma_ST = D_sigma_ST * (~hit_barrier)

        # Calculate Greeks
        Delta = discount_factor * torch.mean(payoff * Dt_ST / (sigma * S0))
        Gamma = discount_factor * torch.mean(payoff * D2t_ST / (sigma**2 * S0**2))
        Vega = discount_factor * torch.mean(payoff * D_sigma_ST / sigma)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item()
        }

    def lookback_option_delta(self, S0, T, r, sigma, num_paths=100000, num_steps=50, seed=42):
        """
        Calculate Delta for Lookback options using Malliavin calculus.

        Args:
            S0 (float): Initial stock price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps in the simulation
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta
        """
        torch.manual_seed(seed)

        dt = T / num_steps
        S = torch.zeros((num_paths, num_steps+1), device=self.device)
        S[:, 0] = S0
        z = torch.randn((num_paths, num_steps), device=self.device)

        # Simulate stock price paths
        for i in range(num_steps):
            S[:, i+1] = S[:, i] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z[:, i])

        # Calculate minimum price and payoff
        S_min, _ = torch.min(S[:, :-1], dim=1)
        payoff = torch.relu(S[:, -1] - S_min)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = S[:, -1] * torch.sum(z, dim=1) / torch.sqrt(dt)
        indicator = (S_min.unsqueeze(1) == S[:, :-1]).float()
        Dt_Smin = S[:, :-1] * z / torch.sqrt(dt)
        Dt_Smin = torch.sum(Dt_Smin * indicator, dim=1) / torch.sum(indicator, dim=1)
        Dt_Smin[torch.isnan(Dt_Smin)] = 0.0

        # Calculate Delta
        Delta = discount_factor * torch.mean((Dt_ST - Dt_Smin) / (sigma * S0) * (payoff > 0))

        return {'Delta': Delta.item()}

    def basket_option_greeks(self, S0, K, T, r, sigma, rho, weights, num_paths=100000, seed=42):
        """
        Calculate Greeks for Basket options using Malliavin calculus.

        Args:
            S0 (list): Initial stock prices
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (list): Volatilities
            rho (list): Correlation matrix
            weights (list): Weights of assets in the basket
            num_paths (int): Number of Monte Carlo paths
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega for each asset
        """
        torch.manual_seed(seed)

        num_assets = len(S0)
        S0 = torch.tensor(S0, device=self.device)
        sigma = torch.tensor(sigma, device=self.device)
        rho = torch.tensor(rho, device=self.device)
        weights = torch.tensor(weights, device=self.device)

        # Cholesky decomposition for correlated random numbers
        L = torch.linalg.cholesky(rho)
        z = torch.randn((num_paths, num_assets), device=self.device)
        z = z @ L.T
        dt = T

        # Simulate stock prices at maturity
        ST = S0 * torch.exp((r - 0.5 * sigma**2) * T + sigma * torch.sqrt(dt) * z)
        basket_price = torch.sum(weights * ST, dim=1)
        payoff = torch.relu(basket_price - K)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = ST * z / torch.sqrt(dt)

        # Calculate Greeks for each asset
        Delta = []
        Gamma = []
        Vega = []
        for i in range(num_assets):
            Delta_i = discount_factor * torch.mean(payoff * weights[i] * Dt_ST[:, i] / (sigma[i] * S0[i]))
            Delta.append(Delta_i.item())

            D2t_ST = Dt_ST[:, i] * (z[:, i] / torch.sqrt(dt) - sigma[i] * torch.sqrt(dt))
            Gamma_i = discount_factor * torch.mean(payoff * weights[i] * D2t_ST / (sigma[i]**2 * S0[i]**2))
            Gamma.append(Gamma_i.item())

            D_sigma_ST = ST[:, i] * (z[:, i] * torch.sqrt(dt) - sigma[i] * dt)
            Vega_i = discount_factor * torch.mean(payoff * weights[i] * D_sigma_ST / sigma[i])
            Vega.append(Vega_i.item())

        return {
            'Delta': Delta,
            'Gamma': Gamma,
            'Vega': Vega
        }

    def asian_option_greeks(self, S0, K, T, r, sigma, num_paths=100000, num_steps=50, seed=42):
        """
        Calculate Greeks for Asian options using Malliavin calculus.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps in the simulation
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega
        """
        torch.manual_seed(seed)

        dt = T / num_steps
        S = torch.zeros((num_paths, num_steps+1), device=self.device)
        S[:, 0] = S0
        z = torch.randn((num_paths, num_steps), device=self.device)

        # Simulate stock price paths
        for i in range(num_steps):
            S[:, i+1] = S[:, i] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z[:, i])

        # Calculate average price and payoff
        S_mean = torch.mean(S[:, 1:], dim=1)
        payoff = torch.relu(S_mean - K)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_S = S[:, 1:] * z / torch.sqrt(dt)
        Dt_S_mean = torch.mean(Dt_S, dim=1)
        D2t_S = Dt_S * (z / torch.sqrt(dt) - sigma * torch.sqrt(dt))
        D2t_S_mean = torch.mean(D2t_S, dim=1)
        D_sigma_S = S[:, 1:] * (z * torch.sqrt(dt) - sigma * dt)
        D_sigma_S_mean = torch.mean(D_sigma_S, dim=1)

        # Calculate Greeks
        Delta = discount_factor * torch.mean(payoff * Dt_S_mean / (sigma * S0))
        Gamma = discount_factor * torch.mean(payoff * D2t_S_mean / (sigma**2 * S0**2))
        Vega = discount_factor * torch.mean(payoff * D_sigma_S_mean / sigma)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item()
        }
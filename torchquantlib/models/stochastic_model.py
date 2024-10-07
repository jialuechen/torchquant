import torch

class StochasticModel:
    """
    Base class for stochastic models.

    This class provides a foundation for implementing various stochastic models
    in financial mathematics, such as stock price models, interest rate models,
    or volatility models.
    """

    def __init__(self, params):
        """
        Initialize the stochastic model.

        Args:
            params (dict): A dictionary of model parameters. Each parameter should be a PyTorch tensor
                           with requires_grad=True if it needs to be optimized.
        """
        self.params = params  # Dictionary of model parameters

        # Determine the device (CPU or GPU) and move all parameters to it
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for param in self.params.values():
            param.data = param.data.to(self.device)

    def simulate(self, S0, T, N, **kwargs):
        """
        Simulate asset prices, interest rates, or other stochastic processes.

        This method should be implemented by subclasses to perform the actual simulation
        based on the specific stochastic model.

        Args:
            S0 (float): Initial value of the process.
            T (float): Time horizon for the simulation.
            N (int): Number of paths to simulate.
            **kwargs: Additional keyword arguments specific to each model.

        Returns:
            Should return a tensor of simulated values.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("simulate method must be implemented by subclasses.")

    def _apply_constraints(self):
        """
        Apply constraints to model parameters after each optimization step.

        This method can be overridden in subclasses to implement specific constraints
        on model parameters, such as ensuring positivity or specific ranges.

        Example implementation in a subclass:
        def _apply_constraints(self):
            self.params['volatility'].data.clamp_(min=0)  # Ensure volatility is non-negative
        """
        pass
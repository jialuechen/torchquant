import torch

class StochasticModel:
    """
    Base class for stochastic models.
    """
    def __init__(self, params):
        self.params = params  # Dictionary of model parameters
        # Move all parameters to the device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for param in self.params.values():
            param.data = param.data.to(self.device)

    def simulate(self, S0, T, N, **kwargs):
        """
        Simulate asset prices or interest rates.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("simulate method must be implemented by subclasses.")

    def _apply_constraints(self):
        """
        Apply parameter constraints after each optimizer step.
        Override in subclasses if needed.
        """
        pass
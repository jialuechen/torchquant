from geomloss import SamplesLoss  # Ensure you have installed geomloss: pip install geomloss
import torch
import torch.optim as optim

class ModelCalibrator:
    """
    Calibrates a stochastic model using various loss functions, including Sinkhorn divergence, MSE, and Log-Likelihood.
    """
    def __init__(self, model, observed_data, S0=None, T=1.0, loss_type="sinkhorn", p=2, blur=0.05, optimizer_cls=optim.Adam, lr=0.01):
        """
        Initialize the calibrator.

        Args:
            model: The stochastic model to calibrate.
            observed_data: Observed data to calibrate against.
            S0: Initial state of the model (optional).
            T: Time horizon for simulation.
            loss_type: Type of loss function ('sinkhorn', 'mse', 'log_likelihood').
            p: Power parameter for Sinkhorn divergence.
            blur: Blur parameter for Sinkhorn divergence.
            optimizer_cls: Optimizer class (default: Adam).
            lr: Learning rate for the optimizer.
        """
        self.model = model
        self.observed_data = torch.tensor(observed_data, dtype=torch.float32, device=self.model.device)
        self.S0 = S0
        self.T = T
        self.loss_type = loss_type
        self.lr = lr
        self.optimizer_cls = optimizer_cls

        # Initialize the loss function
        if loss_type == "sinkhorn":
            self.loss_fn = SamplesLoss(loss_type, p=p, blur=blur)
        elif loss_type == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif loss_type == "log_likelihood":
            self.loss_fn = self._log_likelihood_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self._setup_optimizer()

    def _setup_optimizer(self):
        """Set up the optimizer for model parameters."""
        params = []
        for param in self.model.params.values():
            if param.requires_grad:
                params.append(param)
        self.optimizer = self.optimizer_cls(params, lr=self.lr)

    def _log_likelihood_loss(self, simulated_data, observed_data):
        """
        Compute the negative log-likelihood loss.

        Args:
            simulated_data: Simulated data from the model.
            observed_data: Observed data to calibrate against.

        Returns:
            Tensor: Negative log-likelihood loss.
        """
        # Assume Gaussian likelihood for simplicity
        mean = simulated_data.mean(dim=0)
        variance = simulated_data.var(dim=0) + 1e-6  # Add small value to avoid division by zero
        log_likelihood = -0.5 * torch.sum((observed_data - mean) ** 2 / variance + torch.log(variance))
        return -log_likelihood

    def calibrate(self, num_epochs=1000, batch_size=None, steps=100, verbose=True):
        """
        Perform calibration by minimizing the loss function.

        Args:
            num_epochs: Number of epochs for calibration.
            batch_size: Batch size for simulation (default: size of observed data).
            steps: Number of time steps in the simulation.
            verbose: Whether to print progress during calibration.
        """
        if batch_size is None:
            batch_size = len(self.observed_data)
        observed_tensor = self.observed_data

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            simulated_data = self.model.simulate(S0=self.S0, T=self.T, N=batch_size, steps=steps)

            # Flatten data if necessary
            if simulated_data.dim() > 1:
                simulated_flat = simulated_data.view(batch_size, -1)
                observed_flat = observed_tensor.view(batch_size, -1)
            else:
                simulated_flat = simulated_data.view(-1, 1)
                observed_flat = observed_tensor.view(-1, 1)

            # Compute loss
            loss = self.loss_fn(simulated_flat, observed_flat)
            loss.backward()
            self.optimizer.step()

            # Apply parameter constraints
            self.model._apply_constraints()

            if verbose and epoch % (num_epochs // 10) == 0:
                params_str = ', '.join([f'{name}: {param.item() if param.dim()==0 else param.detach().cpu().numpy()}' for name, param in self.model.params.items()])
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, {params_str}')

    def get_calibrated_params(self):
        """
        Retrieve the calibrated parameters of the model.

        Returns:
            dict: Dictionary of parameter names and their calibrated values.
        """
        result = {}
        for name, param in self.model.params.items():
            if param.dim() == 0:
                result[name] = param.item()
            else:
                result[name] = param.detach().cpu().numpy()
        return result
from geomloss import SamplesLoss  # Ensure you have installed geomloss: pip install geomloss
import torch
import torch.optim as optim

class ModelCalibrator:
    """
    Calibrates a stochastic model using the Sinkhorn divergence.
    """
    def __init__(self, model, observed_data, S0=None, T=1.0, loss_type="sinkhorn", p=2, blur=0.05, optimizer_cls=optim.Adam, lr=0.01):
        self.model = model
        self.observed_data = torch.tensor(observed_data, dtype=torch.float32, device=self.model.device)
        self.S0 = S0
        self.T = T
        self.loss_fn = SamplesLoss(loss_type, p=p, blur=blur)
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self._setup_optimizer()

    def _setup_optimizer(self):
        # Collect parameters that require gradients
        params = []
        for param in self.model.params.values():
            if param.requires_grad:
                params.append(param)
        self.optimizer = self.optimizer_cls(params, lr=self.lr)

    def calibrate(self, num_epochs=1000, batch_size=None, steps=100, verbose=True):
        if batch_size is None:
            batch_size = len(self.observed_data)
        observed_tensor = self.observed_data

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            simulated_data = self.model.simulate(S0=self.S0, T=self.T, N=batch_size, steps=steps)
            if simulated_data.dim() > 1:
                simulated_flat = simulated_data.view(batch_size, -1)
                observed_flat = observed_tensor.view(batch_size, -1)
            else:
                simulated_flat = simulated_data.view(-1, 1)
                observed_flat = observed_tensor.view(-1, 1)

            loss = self.loss_fn(simulated_flat, observed_flat)
            loss.backward()
            self.optimizer.step()

            # Apply parameter constraints
            self.model._apply_constraints()

            if verbose and epoch % (num_epochs // 10) == 0:
                params_str = ', '.join([f'{name}: {param.item() if param.dim()==0 else param.detach().cpu().numpy()}' for name, param in self.model.params.items()])
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, {params_str}')

    def get_calibrated_params(self):
        result = {}
        for name, param in self.model.params.items():
            if param.dim() == 0:
                result[name] = param.item()
            else:
                result[name] = param.detach().cpu().numpy()
        return result
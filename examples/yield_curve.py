import torch
from torchquant.models.yield_curve.yield_curve_construction import bootstrap_yield_curve, nelson_siegel_yield_curve

cash_flows = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])
prices = torch.tensor([0.95, 0.96, 0.97, 0.98, 0.99])
yields = bootstrap_yield_curve(cash_flows, prices)
print(f'Bootstrapped Yields: {yields}')

tau = torch.tensor([0.5, 1.0, 1.5, 2.0])
beta0 = torch.tensor(0.03)
beta1 = torch.tensor(-0.02)
beta2 = torch.tensor(0.02)
yields_ns = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
print(f'Nelson-Siegel Yields: {yields_ns}')
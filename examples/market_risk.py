import torch
from torchquant.risk_management.market_risk.var import calculate_var
from torchquant.risk_management.market_risk.expected_shortfall import calculate_es

returns = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.04, -0.03])
confidence_level = 0.95

var = calculate_var(returns, confidence_level)
es = calculate_es(returns, confidence_level)

print(f'VaR: {var.item()}')
print(f'Expected Shortfall: {es.item()}')
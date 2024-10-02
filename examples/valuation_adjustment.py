import torch
from torchquantlib.core.risk.valuation_adjustments.cva import calculate_cva
from torchquantlib.core.risk.valuation_adjustments.dva import calculate_dva
from torchquantlib.core.risk.valuation_adjustments.mva import calculate_mva
from torchquantlib.core.risk.valuation_adjustments.fva import calculate_fva

exposure = torch.tensor(100.0)
default_prob = torch.tensor(0.02)
recovery_rate = torch.tensor(0.4)
funding_cost = torch.tensor(0.01)
funding_spread = torch.tensor(0.005)
maturity = torch.tensor(1.0)

cva = calculate_cva(exposure, default_prob, recovery_rate)
dva = calculate_dva(exposure, default_prob, recovery_rate)
mva = calculate_mva(exposure, funding_cost, maturity)
fva = calculate_fva(exposure, funding_spread, maturity)

print(f'CVA: {cva.item()}')
print(f'DVA: {dva.item()}')
print(f'MVA: {mva.item()}')
print(f'FVA: {fva.item()}')
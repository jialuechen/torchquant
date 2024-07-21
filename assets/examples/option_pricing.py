from quantorch.core.optionPricer import OptionPricer
from torch import Tensor

optionType='european'
optionDirection='put'
spot=Tensor([100,95]),
strike=Tensor([120,80])
expiry=Tensor([1.0,0.75]),
volatility=Tensor([0.1,0.3])
rate=Tensor([0.01,0.05]),
dividend=Tensor([0.01,0.02])

pricingModel='BSM'

# here we use GPU accleration as an example

if torch.cuda.is_available():
   device=torch.device("cuda")
   spot=spot.to(device)
   strike=strike.to(device)
   expiry=expiry.to(device)
   volatility=volatility.to(device)
   rate=rate.to(device)

# call the forward function in BSM pricing model
OptionPricer.price(
    optionType,
    optionDirection,
    spot,
    strike,
    expiry,
    volatility,
    rate,
    dividend,
    pricingModel,
    device='GPU'
    )
from torch import Tensor
from pricer import Pricer
from instruments.derivatives import EuropeanOption,AmericanOption

class OptionPricer(Pricer):
    
    def __init__(self,optionType='european',pricingModel='BSM'):
        self.option=self.createOption(optionType)
        self.pricingModel=pricingModel
    
    def createOption(self,spot:Tensor,strike:Tensor,expiry:Tensor,rate:Tensor,volatility,optionType='european')->EuropeanOption:
        if optionType=='american':
            return AmericanOption(spot,strike,rate,expiry)
        return EuropeanOption(spot,strike,expiry,rate,)

    def price(self,spot:Tensor,strike:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor,optionType='european')->Tensor:
        return self.pricingModel.forward(self.createOption(spot,strike,expiry,rate,volatility,))
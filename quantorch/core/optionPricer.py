from torch import Tensor
from pricer import Pricer
from instruments.derivatives import EuropeanOption,AmericanOption
from quantorch.core.models.malliavin import MalliavinGreeks

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
    

    @staticmethod
    def price(optionType, optionDirection, spot, strike, expiry, volatility, rate, dividend, pricingModel, device='CPU'):
        # Existing pricing logic...
        pass

    @staticmethod
    def calculate_greeks(optionType, optionDirection, spot, strike, expiry, volatility, rate, dividend, method='finite_difference', num_paths=10000, num_steps=252):
        if method.lower() == 'malliavin':
            malliavin = MalliavinGreeks(None, spot, strike, rate, volatility, expiry, num_paths, num_steps)
            return malliavin.calculate_all_greeks()
        else:
            # Implement finite difference method or other methods here
            pass
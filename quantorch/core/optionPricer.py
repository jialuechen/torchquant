from pricer import Pricer
from instruments.derivatives import EuropeanOption,AmericanOption

class OptionPricer(Pricer):
    
    def __init__(self,optionType='european',pricingModel='BSM'):
        self.option=self.createOption(optionType)
        self.pricingModel=pricingModel

    def createOption(self,optionType='european',spotPrice:Float,strike:float,expiry:float,rate:float)
        if optionType=='american':
            return AmericanOption(spotPrice,strike,rate,expiry)
        return EuropeanOption(spotPrice,strike,rate,expiry)

    def price(self,optionType:'european',spotPrice:float,expiry:float,strike:float,rate:float):
        return self.pricingModel.forward(self.option)
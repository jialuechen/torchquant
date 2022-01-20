from quantorch.black_scholes.blackscholes import EuropeanOption

from enum import Enum

class OptionTypes(Enum):

    @classmethod
    def get_class(cls,style):
        return{cls.EUROPEAN:EuropeanOption}[cls(style)]

class Option:
    def __init__(self) -> None:
        self.__class__=OptionTypes.get_class(style)
        self.__init__(self,*args,**kwargs)

class EuropeanOption:
    def __init__(self,underlying,strike=1.0,is_call=True,maturity=1.0):
        self.underlying=underlying
        self.strike=strike
        self.is_call=is_call
        self.maturity=maturity
    
    def payoff(self,keepdim=False):
        return european_payoff(self.underlying,keepdim=keepdim)

class AmericanOption:
    pass

class AsianOption:
    pass

class binaryOption:
    pass


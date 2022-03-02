import imp
from quantorch.black_scholes.blackscholes import EuropeanOption

from enum import Enum

from utils import calculate_payoff

class OptionTypes(Enum):

    @classmethod
    def get_class(cls,style):
        return{cls.EUROPEAN:EuropeanOption}[cls(style)]

class Option:

    def __init__(self) -> None:
        self.__class__=OptionTypes.get_class(style)
        self.__init__(self,*args,**kwargs)

class EuropeanOption:
    def __init__(self,underlying_price,strike:float=1.0,is_call:bool=True,dividend:float=0,maturity:float=1.0):
        self.underlying_price=underlying_price
        self.strike=strike
        self.is_call=is_call
        self.dividend=dividend
        self.maturity=maturity
    
    def payoff(self,underlying_price,is_call,strike):
        return calculate_payoff(underlying_price=underlying_price,strike=strike, is_call=is_call,optionType='european')

class AmericanOption:
    pass

class AsianOption:
    pass

class BermudanOption:
    pass

class lookbackOption:
    pass

class binaryOption:
    pass



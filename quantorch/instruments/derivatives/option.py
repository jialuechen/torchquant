from quantorch.black_scholes.blackscholes import EuropeanOption

from enum import Enum

from utils import calculate_payoff

class OptionTypes(Enum):

    @classmethod
    def get_class(cls,category):
        return{cls.EUROPEAN:EuropeanOption}[cls(category)]

class Option:

    def __init__(self,category,*args,**kwargs) -> None:
        self.__class__=OptionTypes.get_class(category)
        self.__init__(self,*args,**kwargs)

class EuropeanOption:
    def __init__(self,underlying_price,strike:float=1.0,is_call:bool=True,dividend:float=0,risk_free_rate:float=0.0,maturity:float=1.0):
        self.underlying_price=underlying_price
        self.strike=strike
        self.is_call=is_call
        self.dividend=dividend
        self.risk_free_rate=risk_free_rate
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



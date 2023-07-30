from enum import Enum
from torch import Tensor

class OptionTypes(Enum):

    @classmethod
    def get_class(cls,category):
        return{cls.EUROPEAN:BSM_EuropeanOption}[cls(category)]

class Option:

    def __init__(self,category,*args,**kwargs) -> None:
        self.__class__=OptionTypes.get_class(category)
        self.__init__(self,*args,**kwargs)

class EuropeanOption:
    def __init__(self,spot:Tensor,strike:Tensor,is_call:bool=True,dividend:float=0,risk_free_rate:float=0.0,expiry:float=1.0):
        self.underlying_price=spot
        self.strike=strike
        self.is_call=is_call
        self.dividend=dividend
        self.risk_free_rate=risk_free_rate
        self.expiry=expiry
    '''
    def payoff(self,spot:Tensor,is_call,strike:Tensor):
        return calculate_payoff(spot=spot,strike=strike, is_call=is_call,optionType='european')
    '''

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



import imp
import torch
from torch import Tensor
from bsm import BSM

class EuropeanOption(BSM):
    
    def __init__(self,is_call=True):
        self.is_call=is_call
        
    def price(self,underlying_price:Tensor,strike:Tensor,volatility:Tensor,maturity:Tensor,risk_free_rate:Tensor,dividend:Tensor)->Tensor:
        volatility,spot,strike,maturity=map(torch.as_tensor,(volatility,spot,strike,maturity))
        n1=self.nd(self.d1(volatility,spot,strike,maturity))
        n2=self.nd(self.d2(volatility,spot,strike,maturity))
        price=underlying_price*n1*torch.exp(-maturity*dividend)-strike*n2*torch.exp(-maturity*risk_free_rate)
        """
        Use put-call parity to calculate put option price
        """
        if not self.is_call:
            price=strike*n2-spot*n1
        return price

class AmericanOption(BSM):
    pass
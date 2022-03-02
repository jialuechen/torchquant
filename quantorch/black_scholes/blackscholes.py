import imp
import torch
from torch.distributions.normal import Normal
from torch import Tensor

class BlackScholes:
    
    @property
    def nd(self):
        normal_dist=Normal(torch.tensor(0.0),torch.tensor(1.0))
        setattr(normal_dist,"cdf",lambda input:normal_dist.log_prob(input).exp())
        return normal_dist
    
    @staticmethod
    def d1(volatility:torch.Tensor,strike:float,underlying_price:float,maturity:torch.Tensor)->torch.Tensor:
        sigma,s,k,tau=map(torch.as_tensor,(volatility,underlying_price,strike,maturity))
        return (torch.log(s/k)+(sigma**2/2)*tau)/(sigma*torch.sqrt(tau))
    
    @staticmethod
    def d2(volatility:torch.Tensor,strike:float,underlying_price:float,maturity:torch.Tensor)->torch.Tensor:
        sigma,s,k,tau=map(torch.as_tensor,(volatility,underlying_price,strike,maturity))
        return (torch.log(s/k)-(sigma**2/2)*tau)/(sigma*torch.sqrt(tau))


class BS_EuropeanOption(BlackScholes):
    
    def __init__(self,is_call=True):
        self.is_call=is_call
        
    def price(self,underlying_price,strike,volatility,maturity)->Tensor:
        sigma,s,k,tau=map(torch.as_tensor,(volatility,underlying_price,strike,maturity))
        n1=self.nd(self.d1(sigma,s,k,tau))
        n2=self.nd(self.d2(sigma,s,k,tau))
        price=underlying_price*n1-strike*n2
        """
        Use put-call parity to calculate put option price
        """
        if not self.is_call:
            price=strike*n2-underlying_price*n1
        return price


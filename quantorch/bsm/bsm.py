import torch
from torch.distributions.normal import Normal
from torch import Tensor

class BSM:
    
    @property
    def nd(self):
        normal_dist=Normal(torch.tensor(0.0),torch.tensor(1.0))
        setattr(normal_dist,"cdf",lambda input:normal_dist.log_prob(input).exp())
        return normal_dist
    
    @staticmethod
    def d1(volatility:torch.Tensor,strike:Tensor,spot:Tensor,maturity:Tensor)->Tensor:
        volatility,s,k,maturity=map(torch.as_tensor,(volatility,spot,strike,maturity))
        return (torch.log(spot/strike)+(volatility**2/2)*maturity)/(volatility*torch.sqrt(maturity))
    
    @staticmethod
    def d2(volatility:Tensor,strike:Tensor,spot:Tensor,maturity:Tensor)->Tensor:
        volatility,s,k,maturity=map(torch.as_tensor,(volatility,spot,strike,maturity))
        return (torch.log(s/k)-(volatility**2/2)*maturity)/(volatility*torch.sqrt(maturity))



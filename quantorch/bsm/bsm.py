import torch
from torch.distributions.normal import Normal
from torch import Tensor

class BSM:
    def nd(self):
        normal_dist=Normal(torch.tensor(0.0),torch.tensor(1.0))
        setattr(normal_dist,"cdf",lambda input:normal_dist.log_prob(input).exp())
        return normal_dist
    
    def d1(self,strike:Tensor,spot:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor)->Tensor:
        volatility,s,k,expiry=map(torch.as_tensor,(volatility,spot,strike,expiry))
        return (torch.log(spot/strike)+(volatility**2/2)*expiry)/(volatility*torch.sqrt(expiry))
    
    def d2(self,strike:Tensor,spot:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor)->Tensor:
        volatility,s,k,expiry=map(torch.as_tensor,(volatility,spot,strike,expiry))
        return (torch.log(s/k)-(volatility**2/2)*expiry)/(volatility*torch.sqrt(expiry))
    
    def forward(self,dividend:Tensor,strike:Tensor,spot:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor)->Tensor:
        return spot*self.nd(self.d1(volatility,strike,spot,expiry,rate))*torch.exp(-dividend)-strike*self.nd(self.d2(volatility,strike,spot,expiry,rate))*torch.exp(-rate)


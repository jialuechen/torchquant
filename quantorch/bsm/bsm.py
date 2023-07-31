import torch
from torch.distributions.normal import Normal
from torch import Tensor

class BSM:
    def nd(self):
        normal_dist=Normal(torch.tensor(0.0),torch.tensor(1.0))
        setattr(normal_dist,"cdf",lambda input:normal_dist.log_prob(input).exp())
        return normal_dist
    
    def d1(self,strike:Tensor,spot:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor,dividend:Tensor)->Tensor:
        return (torch.log(spot/strike)+(rate-dividend+volatility**2/2)*expiry)/(volatility*torch.sqrt(expiry))
    
    def d2(self,strike:Tensor,spot:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor,dividend:Tensor)->Tensor:
        return (torch.log(spot/strike)-(rate-dividend-volatility**2/2)*expiry)/(volatility*torch.sqrt(expiry))
    
    def forward(self,isCall:True,strike:Tensor,spot:Tensor,expiry:Tensor,rate:Tensor,volatility:Tensor,dividend:Tensor)->Tensor:
        callPrice=spot*self.nd(self.d1(volatility,strike,spot,expiry,rate,dividend))*torch.exp(-dividend*expiry)-\
            strike*self.nd(self.d2(volatility,strike,spot,expiry,rate,dividend))*torch.exp(-rate*expiry)
        if not isCall:
            return strike*torch.exp(-rate*expiry)-spot+callPrice
        return callPrice

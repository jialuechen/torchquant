import torch
from torch.distributions.normal import Normal

class BlackScholes:
    
    @property
    def N(self):
        normal=Normal(torch.tensor(0.0),torch.tensor(1.0))
        setattr(normal,"pdf",lambda input:normal.log_prob(input).exp())
        return normal
    

class EuropeanOption(BlackScholes):
    
    def __init__(self,is_call=True):
        self.is_call=is_call
        
    def price():
        s,t,v=map(torch.as_tensor,(expiry,volatility))
        n1=self.N.cdf()
        price=strike*(torch.exp(s)*n1-n2)
        """
        put-call parity
        """
        if not self.is_call:
            price+=strike*(torch.exp(s)-1)
        return price


from textwrap import wrap
import torch

class Vasicek:
    def __init__(self,alpha,sigma) -> None:
        self.alpha=alpha
        self.sigma=sigma
        self.init_r=wrap(0.0)
    
    def get_params(self):
        return torch.cat([self.alpha,self.sigma])
    
    def step(self,dt,r0,defl0):
        numSim=r0.size(0)

    def simulate(self):
        pass

    def zcb_price(self,r,tenor,params=None):
        pass
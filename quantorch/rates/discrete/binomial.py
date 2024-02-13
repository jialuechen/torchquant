from torch import Tensor,exp

class BinomalTree():
    def __init__(self,period:Tensor,spot:Tensor,u:Tensor,d:Tensor,rate:Tensor,) -> None:
        self.period=period
        self.spot=spot
        self.u=u
        self.d=d
        self.rate=rate

    def getRiskNeutralProb(self)->Tensor:
        return [(exp(self.rate)-self.d)/(self.u-self.d),1-(exp(self.rate)-self.d)/(self.u-self.d)]
    
    def getRiskNeutralDelta(self)->Tensor:
        pass

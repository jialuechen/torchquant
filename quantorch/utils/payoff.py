import imp
from torch import Tensor

def calculate_payoff(underlying_price,strike:float=1.0,is_call=True,optiontype='european'):
    if optiontype=='european':
        if is_call:return max(0.0,underlying_price[-1]-strike)


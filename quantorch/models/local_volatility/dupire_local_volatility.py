import torch
from torch import Tensor
from scipy.interpolate import interp2d
import numpy as np

class DupireLocalVolatility:
    def __init__(self, spots: Tensor, strikes: Tensor, expiries: Tensor, implied_vols: Tensor):
        self.spots = spots
        self.strikes = strikes
        self.expiries = expiries
        self.implied_vols = implied_vols
        self.interpolator = interp2d(strikes.numpy(), expiries.numpy(), implied_vols.numpy(), kind='cubic')

    def local_vol(self, spot: float, strike: float, expiry: float) -> float:
        imp_vol = self.interpolator(strike, expiry)[0]
        d1 = (torch.log(spot / strike) + (0.5 * imp_vol**2) * expiry) / (imp_vol * torch.sqrt(expiry))
        vega = spot * torch.sqrt(expiry) * torch.exp(-0.5 * d1**2) / torch.sqrt(2 * torch.pi)
        local_vol = imp_vol * torch.sqrt(1 + 2 * d1 * imp_vol * torch.sqrt(expiry) / vega)
        return local_vol
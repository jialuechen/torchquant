import unittest
from quantorch.core.optionPricer import OptionPricer
from torch import Tensor

class TestOptionPricer(unittest.TestCase):
    
    def test_bsm_pricing(self):
        spot = Tensor([100])
        strike = Tensor([120])
        expiry = Tensor([1.0])
        volatility = Tensor([0.1])
        rate = Tensor([0.01])
        dividend = Tensor([0.01])
        price = OptionPricer.price(
            optionType='european',
            optionDirection='put',
            spot=spot,
            strike=strike,
            expiry=expiry,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
            pricingModel='BSM',
            device='cpu'
        )
        self.assertIsNotNone(price, "Price should not be None")

if __name__ == '__main__':
    unittest.main()
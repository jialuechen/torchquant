Black-Scholes-Merton Model
==========================

.. automodule:: torchquantlib.core.asset_pricing.option.black_scholes_merton
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Black-Scholes-Merton module implements the Black-Scholes-Merton model for pricing European and American options on dividend-paying stocks.

Key Features
------------

- Pricing of European and American call and put options
- Support for dividend-paying stocks
- Efficient implementation using PyTorch tensors

Functions
---------

.. autofunction:: torchquantlib.core.asset_pricing.option.black_scholes_merton.black_scholes_merton

   Price an option using the Black-Scholes-Merton model.

   :param option_type: Type of option - either 'call' or 'put'
   :type option_type: str
   :param option_style: Style of option - either 'european' or 'american'
   :type option_style: str
   :param spot: Current price of the underlying asset
   :type spot: Tensor
   :param strike: Strike price of the option
   :type strike: Tensor
   :param expiry: Time to expiration in years
   :type expiry: Tensor
   :param volatility: Volatility of the underlying asset
   :type volatility: Tensor
   :param rate: Risk-free interest rate (annualized)
   :type rate: Tensor
   :param dividend: Continuous dividend yield of the underlying asset
   :type dividend: Tensor
   :return: The price of the option
   :rtype: Tensor

Examples
--------

Here's a basic example of how to use the black_scholes_merton function:

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton

   # Set option parameters
   option_type = 'call'
   option_style = 'european'
   spot = torch.tensor(100.0)
   strike = torch.tensor(95.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   dividend = torch.tensor(0.02)

   # Price the option
   price = black_scholes_merton(option_type, option_style, spot, strike, expiry, volatility, rate, dividend)
   print(f"Black-Scholes-Merton {option_style} {option_type} option price: {price:.4f}")
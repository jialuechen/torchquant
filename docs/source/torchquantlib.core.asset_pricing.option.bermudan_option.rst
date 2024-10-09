Bermudan Option
===============

.. automodule:: torchquantlib.core.asset_pricing.option.bermudan_option
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Bermudan Option module provides functionality for pricing Bermudan-style options using a binomial tree model. Bermudan options are a type of exotic option that can be exercised on specific dates before expiration, combining features of both American and European options.

Key Features
------------

- Pricing of Bermudan call and put options
- Implementation using a binomial tree model
- Support for multiple exercise dates

Functions
---------

.. autofunction:: torchquantlib.core.asset_pricing.option.bermudan_option.bermudan_option

   Price a Bermudan option using a binomial tree model.

   :param option_type: Type of option - either 'call' or 'put'
   :type option_type: str
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
   :param steps: Number of time steps in the binomial tree
   :type steps: int
   :param exercise_dates: Tensor of indices representing the steps at which the option can be exercised
   :type exercise_dates: Tensor
   :return: The price of the Bermudan option
   :rtype: Tensor

Examples
--------

Here's a basic example of how to use the bermudan_option function:

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.bermudan_option import bermudan_option

   # Set option parameters
   option_type = 'call'
   spot = torch.tensor(100.0)
   strike = torch.tensor(95.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   steps = 100
   exercise_dates = torch.tensor([25, 50, 75])  # Exercise allowed at 1/4, 1/2, and 3/4 of the option's life

   # Price the Bermudan option
   price = bermudan_option(option_type, spot, strike, expiry, volatility, rate, steps, exercise_dates)
   print(f"Bermudan {option_type} option price: {price:.4f}")
Exotic Options
==============

.. automodule:: torchquantlib.core.asset_pricing.option.exotics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Exotic Options module provides functionality for pricing various types of exotic options, including barrier options, chooser options, compound options, shout options, and lookback options.

Key Features
------------

- Pricing of multiple exotic option types
- Implementation of various numerical methods for option pricing
- Support for PyTorch tensors for efficient computation

Functions
---------

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.barrier_option

   Price a barrier option using a binomial tree model.

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.chooser_option

   Calculate the price of a chooser option.

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.compound_option

   Calculate the price of a compound option (an option on an option).

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.shout_option

   Calculate the price of a shout option.

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.lookback_option

   Price a lookback option using a binomial tree model.

Examples
--------

Here's a basic example of how to use the barrier_option function:

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.exotics import barrier_option

   # Set option parameters
   option_type = 'call'
   barrier_type = 'up-and-out'
   spot = torch.tensor(100.0)
   strike = torch.tensor(95.0)
   barrier = torch.tensor(120.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   steps = 100

   # Price the barrier option
   price = barrier_option(option_type, barrier_type, spot, strike, barrier, expiry, volatility, rate, steps)
   print(f"Barrier option price: {price:.4f}")
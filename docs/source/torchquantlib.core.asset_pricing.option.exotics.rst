Exotic Options
==============

.. automodule:: torchquantlib.core.asset_pricing.option.exotics
   :members:
   :undoc-members:
   :show-inheritance:

This module provides implementations for various exotic options pricing models.

Barrier Option
--------------

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.barrier_option

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.exotics import barrier_option

   # Example inputs
   option_type = 'call'
   barrier_type = 'up-and-out'
   spot = torch.tensor(100.0)
   strike = torch.tensor(110.0)
   barrier = torch.tensor(120.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   steps = 100

   # Calculate barrier option price
   price = barrier_option(option_type, barrier_type, spot, strike, barrier, expiry, volatility, rate, steps)
   print(f"Barrier option price: {price.item():.4f}")

Chooser Option
--------------

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.chooser_option

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.exotics import chooser_option

   # Example inputs
   spot = torch.tensor(100.0)
   strike = torch.tensor(100.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   dividend = torch.tensor(0.02)

   # Calculate chooser option price
   price = chooser_option(spot, strike, expiry, volatility, rate, dividend)
   print(f"Chooser option price: {price.item():.4f}")

Compound Option
---------------

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.compound_option

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.exotics import compound_option

   # Example inputs
   spot = torch.tensor(100.0)
   strike1 = torch.tensor(110.0)
   strike2 = torch.tensor(10.0)
   expiry1 = torch.tensor(1.0)
   expiry2 = torch.tensor(0.5)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   dividend = torch.tensor(0.02)

   # Calculate compound option price
   price = compound_option(spot, strike1, strike2, expiry1, expiry2, volatility, rate, dividend)
   print(f"Compound option price: {price.item():.4f}")

Shout Option
------------

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.shout_option

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.exotics import shout_option

   # Example inputs
   spot = torch.tensor(100.0)
   strike = torch.tensor(100.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   dividend = torch.tensor(0.02)

   # Calculate shout option price
   price = shout_option(spot, strike, expiry, volatility, rate, dividend)
   print(f"Shout option price: {price.item():.4f}")

Lookback Option
---------------

.. autofunction:: torchquantlib.core.asset_pricing.option.exotics.lookback_option

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.exotics import lookback_option

   # Example inputs
   option_type = 'call'
   spot = torch.tensor(100.0)
   strike = torch.tensor(100.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   steps = 100

   # Calculate lookback option price
   price = lookback_option(option_type, spot, strike, expiry, volatility, rate, steps)
   print(f"Lookback option price: {price.item():.4f}")
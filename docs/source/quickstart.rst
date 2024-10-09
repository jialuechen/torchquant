Quickstart Guide
================

This guide will help you get started with TorchQuant quickly.

Basic Usage
-----------

Here's a simple example of using TorchQuant:

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton

   spot = torch.tensor(100.0)
   strike = torch.tensor(105.0)
   expiry = torch.tensor(1.0)
   volatility = torch.tensor(0.2)
   rate = torch.tensor(0.05)
   dividend = torch.tensor(0.02)

  price = black_scholes_merton('call', 'european', spot, strike, expiry, volatility, rate, dividend)
  print(f'Option Price: {price.item()}')

This example demonstrates how to use the Black-Scholes model to price a European call option.
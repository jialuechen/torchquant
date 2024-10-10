Future Pricer
=============

.. automodule:: torchquantlib.core.asset_pricing.money.future_pricer
   :members:
   :undoc-members:
   :show-inheritance:

This module provides implementation for pricing futures contracts.


.. autofunction:: torchquantlib.core.asset_pricing.money.future_pricer.future_pricer

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.money.future_pricer import future_pricer

   spot = torch.tensor(100.0)
   rate = torch.tensor(0.05)
   expiry = torch.tensor(1.0)

   futures_price = future_pricer(spot, rate, expiry)
   print(f"Futures Price: {futures_price.item():.2f}")

Formula
^^^^^^^

The theoretical price of a futures contract is calculated using the following formula:

.. math::

   Futures Price = Spot Price * e^{(risk-free rate * time to expiry)}

Where:
   - Spot Price is the current market price of the underlying asset
   - risk-free rate is the annualized risk-free interest rate
   - time to expiry is the time until the futures contract expires (in years)

Note:
   - This model assumes perfect markets with no transaction costs or taxes.
   - For commodities or dividend-paying stocks, additional factors would need to be considered in the pricing model.
   - The formula uses continuous compounding.

Limitations
^^^^^^^^^^^

The future pricer implemented in this module has some limitations:

1. It assumes continuous compounding, which may not always reflect real-world scenarios.
2. It does not account for dividends, which can affect the futures price for stock index futures.
3. It does not consider storage costs or convenience yields, which are important factors for commodity futures.
4. The model assumes perfect markets without transaction costs, taxes, or other frictions.

For more complex scenarios or specific types of futures contracts, you may need to modify the pricing model or use more advanced techniques.
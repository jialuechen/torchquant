Fixed Income Forward
====================

.. automodule:: torchquantlib.core.asset_pricing.bond.fixed_income_forward
   :members:
   :undoc-members:
   :show-inheritance:

This module provides implementation for pricing fixed income forward contracts.

Fixed Income Forward
--------------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.fixed_income_forward.fixed_income_forward

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.bond.fixed_income_forward import fixed_income_forward

   face_value = torch.tensor(1000.0)
   rate = torch.tensor(0.05)
   time_to_maturity = torch.tensor(2.0)
   forward_rate = torch.tensor(0.06)

   forward_price = fixed_income_forward(face_value, rate, time_to_maturity, forward_rate)
   print(f"Fixed Income Forward Price: {forward_price.item():.2f}")

Formula
^^^^^^^

The forward price of a fixed income security is calculated using the following formula:

.. math::

   Forward Price = Face Value * e^{(Forward Rate - Current Rate) * Time to Maturity}

Where:
   - Face Value is the notional amount of the fixed income security
   - Forward Rate is the interest rate agreed upon for the forward contract
   - Current Rate is the current market interest rate
   - Time to Maturity is the time until the forward contract expires (in years)

Note that this formula assumes continuous compounding. For discrete compounding, a different formula would be required.
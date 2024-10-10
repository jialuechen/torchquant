Bond Pricer
===========

.. automodule:: torchquantlib.core.asset_pricing.bond.bond_pricer
   :members:
   :undoc-members:
   :show-inheritance:

This module provides implementations for various bond pricing models.

Zero Coupon Bond
----------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.zero_coupon_bond

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.bond.bond_pricer import zero_coupon_bond

   face_value = torch.tensor(1000.0)
   rate = torch.tensor(0.05)
   maturity = torch.tensor(5.0)

   price = zero_coupon_bond(face_value, rate, maturity)
   print(f"Zero Coupon Bond Price: {price.item():.2f}")

Coupon Bond
-----------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.coupon_bond

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from torchquantlib.core.asset_pricing.bond.bond_pricer import coupon_bond

   face_value = torch.tensor(1000.0)
   coupon_rate = torch.tensor(0.06)
   rate = torch.tensor(0.05)
   periods = torch.tensor(10)

   price = coupon_bond(face_value, coupon_rate, rate, periods)
   print(f"Coupon Bond Price: {price.item():.2f}")

Callable Bond
-------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.callable_bond

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from torchquantlib.core.asset_pricing.bond.bond_pricer import callable_bond

   face_value = torch.tensor(1000.0)
   coupon_rate = torch.tensor(0.06)
   rate = torch.tensor(0.05)
   periods = torch.tensor(10)
   call_price = torch.tensor(1050.0)
   call_period = torch.tensor(5)

   price = callable_bond(face_value, coupon_rate, rate, periods, call_price, call_period)
   print(f"Callable Bond Price: {price.item():.2f}")

Putable Bond
------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.putable_bond

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from torchquantlib.core.asset_pricing.bond.bond_pricer import putable_bond

   face_value = torch.tensor(1000.0)
   coupon_rate = torch.tensor(0.06)
   rate = torch.tensor(0.05)
   periods = torch.tensor(10)
   put_price = torch.tensor(950.0)
   put_period = torch.tensor(5)

   price = putable_bond(face_value, coupon_rate, rate, periods, put_price, put_period)
   print(f"Putable Bond Price: {price.item():.2f}")

Convertible Bond
----------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.convertible_bond

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from torchquantlib.core.asset_pricing.bond.bond_pricer import convertible_bond

   face_value = torch.tensor(1000.0)
   coupon_rate = torch.tensor(0.06)
   rate = torch.tensor(0.05)
   periods = torch.tensor(10)
   conversion_ratio = torch.tensor(20)
   conversion_price = torch.tensor(55.0)

   price = convertible_bond(face_value, coupon_rate, rate, periods, conversion_ratio, conversion_price)
   print(f"Convertible Bond Price: {price.item():.2f}")

Stochastic Rate Bond
--------------------

.. autofunction:: torchquantlib.core.asset_pricing.bond.bond_pricer.stochastic_rate_bond

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.core.asset_pricing.bond.bond_pricer import stochastic_rate_bond

   face_value = torch.tensor(1000.0)
   coupon_rate = torch.tensor(0.06)
   rate = torch.tensor([0.05, 0.052, 0.054, 0.056, 0.058])
   periods = torch.tensor(5)

   price = stochastic_rate_bond(face_value, coupon_rate, rate, periods)
   print(f"Stochastic Rate Bond Price: {price.item():.2f}")
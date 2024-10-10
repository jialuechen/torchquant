Valuation Adjustment API
========================

.. currentmodule:: torchquantlib.core.risk.valuation_adjustment.valuation_adjustment

This module provides functions for calculating various valuation adjustments in financial risk management.

Functions
---------

.. autofunction:: calculate_cva

.. autofunction:: calculate_dva

.. autofunction:: calculate_fva

.. autofunction:: calculate_mva

Detailed Description
--------------------

The valuation adjustment module offers a set of functions to compute different types of valuation adjustments commonly used in financial risk management and derivatives pricing.

Credit Valuation Adjustment (CVA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `calculate_cva` function computes the Credit Valuation Adjustment, which represents the market value of counterparty credit risk.

.. math::

   CVA = Exposure \times (1 - RecoveryRate) \times DefaultProbability

Debit Valuation Adjustment (DVA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `calculate_dva` function calculates the Debit Valuation Adjustment, which is similar to CVA but represents the credit risk of the entity itself.

.. math::

   DVA = Exposure \times (1 - RecoveryRate) \times DefaultProbability

Funding Valuation Adjustment (FVA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `calculate_fva` function determines the Funding Valuation Adjustment, which represents the cost of funding for uncollateralized derivatives.

.. math::

   FVA = Exposure \times FundingSpread \times Maturity

Margin Valuation Adjustment (MVA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `calculate_mva` function computes the Margin Valuation Adjustment, which represents the cost of posting initial margin for cleared or non-cleared derivatives.

.. math::

   MVA = Exposure \times FundingCost \times Maturity

Usage Example
^^^^^^^^^^^^^

Here's a basic example of how to use the valuation adjustment functions:

.. code-block:: python

    import torch
    from torchquantlib.core.risk.valuation_adjustment.valuation_adjustment import calculate_cva, calculate_dva, calculate_fva, calculate_mva

    # Set up parameters
    exposure = torch.tensor(1000000.0)
    default_prob = torch.tensor(0.05)
    recovery_rate = torch.tensor(0.4)
    funding_spread = torch.tensor(0.02)
    funding_cost = torch.tensor(0.03)
    maturity = torch.tensor(5.0)

    # Calculate adjustments
    cva = calculate_cva(exposure, default_prob, recovery_rate)
    dva = calculate_dva(exposure, default_prob, recovery_rate)
    fva = calculate_fva(exposure, funding_spread, maturity)
    mva = calculate_mva(exposure, funding_cost, maturity)

    print(f"CVA: {cva.item():.2f}")
    print(f"DVA: {dva.item():.2f}")
    print(f"FVA: {fva.item():.2f}")
    print(f"MVA: {mva.item():.2f}")

Note
^^^^

All functions in this module use PyTorch tensors for input and output, allowing for efficient computation and automatic differentiation when needed.
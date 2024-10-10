Expected Shortfall (ES)
=======================

.. currentmodule:: torchquantlib.core.risk.market_risk.expected_shortfall

The Expected Shortfall (ES) module provides functionality to calculate the Expected Shortfall, also known as Conditional Value at Risk (CVaR), for financial risk management.

Functions
---------

.. autofunction:: calculate_es

Usage Example
-------------

Here's an example of how to use the `calculate_es` function:

.. code-block:: python

    import torch
    from torchquantlib.core.risk.market_risk.expected_shortfall import calculate_es

    # Generate sample returns data
    returns = torch.randn(1000)  # 1000 random returns

    # Calculate Expected Shortfall at 95% confidence level
    es_95 = calculate_es(returns, confidence_level=0.95)

    print(f"Expected Shortfall (95% confidence): {es_95.item():.4f}")

    # Calculate Expected Shortfall at 99% confidence level
    es_99 = calculate_es(returns, confidence_level=0.99)

    print(f"Expected Shortfall (99% confidence): {es_99.item():.4f}")

Notes
-----

- Expected Shortfall is always greater than or equal to Value at Risk (VaR) for the same confidence level.
- ES provides a more comprehensive view of tail risk compared to VaR.
- Higher confidence levels result in more conservative (higher) ES estimates.
- The `calculate_es` function internally uses the `calculate_var` function from the VaR module.

See Also
--------

- :doc:`var` for information on Value at Risk calculation.
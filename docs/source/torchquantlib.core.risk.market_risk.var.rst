Value at Risk (VaR)
===================

.. currentmodule:: torchquantlib.core.risk.market_risk.var
   

The Value at Risk (VaR) module provides functionality to calculate VaR, a widely used measure of financial risk.

Functions
---------

.. autofunction:: calculate_var

Usage Example
-------------

Here's an example of how to use the `calculate_var` function:

.. code-block:: python

    import torch
    from torchquantlib.core.risk.market_risk.var import calculate_var

    # Generate sample returns data
    returns = torch.randn(1000)  # 1000 random returns

    # Calculate VaR at 95% confidence level
    var_95 = calculate_var(returns, confidence_level=0.95)

    print(f"Value at Risk (95% confidence): {var_95.item():.4f}")

    # Calculate VaR at 99% confidence level
    var_99 = calculate_var(returns, confidence_level=0.99)

    print(f"Value at Risk (99% confidence): {var_99.item():.4f}")

Notes
-----

- VaR represents the maximum potential loss at a given confidence level over a specific time horizon.
- A higher confidence level results in a more conservative (higher) VaR estimate.
- VaR is widely used but has limitations, especially in capturing tail risks.
- Consider using Expected Shortfall (ES) alongside VaR for a more comprehensive risk assessment.

See Also
--------

- :doc:`expected_shortfall` for information on Expected Shortfall calculation, which addresses some limitations of VaR.
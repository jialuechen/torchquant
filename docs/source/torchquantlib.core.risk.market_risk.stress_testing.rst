Stress Testing
==============

.. currentmodule:: torchquantlib.core.risk.market_risk.stress_testing

The Stress Testing module provides tools for assessing the impact of extreme market conditions on financial portfolios or instruments.

Functions
---------

.. autofunction:: perform_stress_test

Usage Example
-------------

Here's an example of how to use the stress testing functionality:

.. code-block:: python

    import torch
    from torchquantlib.core.risk.market_risk.stress_testing import perform_stress_test

    # Define a sample portfolio
    portfolio = torch.tensor([100000.0, 50000.0, 75000.0])  # Holdings in three assets

    # Define stress scenarios
    scenarios = {
        "severe_recession": {"asset1": -0.3, "asset2": -0.4, "asset3": -0.25},
        "market_crash": {"asset1": -0.5, "asset2": -0.6, "asset3": -0.55},
        "currency_crisis": {"asset1": -0.2, "asset2": -0.1, "asset3": -0.4}
    }

    # Perform stress test
    results = perform_stress_test(portfolio, scenarios)

    # Print results
    for scenario, impact in results.items():
        print(f"Scenario: {scenario}")
        print(f"Portfolio impact: ${impact:.2f}")
        print(f"Percentage change: {(impact / portfolio.sum().item()) * 100:.2f}%")
        print()

Notes
-----

- Stress testing helps identify potential vulnerabilities in a portfolio under extreme market conditions.
- The scenarios should be carefully chosen to reflect realistic but severe market events.
- Regular stress testing is crucial for robust risk management and regulatory compliance.

See Also
--------

- :doc:`scenario_analysis` for related scenario-based risk assessment techniques.
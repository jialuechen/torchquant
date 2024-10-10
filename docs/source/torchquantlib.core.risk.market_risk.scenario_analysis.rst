Scenario Analysis
=================

.. currentmodule:: torchquantlib.core.risk.market_risk.scenario_analysis

The Scenario Analysis module provides tools for evaluating the impact of various market scenarios on financial portfolios or instruments.

Functions
---------

.. autofunction:: run_scenario_analysis

Usage Example
-------------

Here's an example of how to use the scenario analysis functionality:

.. code-block:: python

    import torch
    from torchquantlib.core.risk.market_risk.scenario_analysis import run_scenario_analysis

    # Define a sample portfolio
    portfolio = torch.tensor([100000.0, 75000.0, 50000.0])  # Holdings in three assets

    # Define scenarios
    scenarios = {
        "base_case": {"asset1": 0.05, "asset2": 0.03, "asset3": 0.04},
        "bull_market": {"asset1": 0.15, "asset2": 0.12, "asset3": 0.10},
        "bear_market": {"asset1": -0.10, "asset2": -0.08, "asset3": -0.12},
        "sector_rotation": {"asset1": -0.05, "asset2": 0.10, "asset3": 0.02}
    }

    # Run scenario analysis
    results = run_scenario_analysis(portfolio, scenarios)

    # Print results
    for scenario, outcome in results.items():
        print(f"Scenario: {scenario}")
        print(f"Portfolio value: ${outcome['portfolio_value']:.2f}")
        print(f"Absolute change: ${outcome['absolute_change']:.2f}")
        print(f"Percentage change: {outcome['percentage_change']:.2f}%")
        print()

Notes
-----

- Scenario analysis helps in understanding the potential outcomes of different market conditions.
- It's important to consider a wide range of scenarios, including both positive and negative outcomes.
- The results can be used for strategic decision-making and risk mitigation planning.

See Also
--------

- :doc:`stress_testing` for assessing the impact of extreme market conditions.
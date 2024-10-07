from torch import Tensor

def stress_test(portfolio_value: Tensor, stress_scenarios: Tensor) -> Tensor:
    """
    Perform stress testing on a portfolio.

    Stress testing is a risk management technique that assesses the potential impact of
    extreme but plausible scenarios on a portfolio's value. It helps identify vulnerabilities
    and estimate potential losses under adverse conditions.

    Args:
        portfolio_value (Tensor): The current value of the portfolio.
        stress_scenarios (Tensor): A tensor of stress scenario percentages, where each element
                                   represents a different stress scenario (e.g., -30% for a market crash).

    Returns:
        Tensor: A tensor of portfolio values under each stress scenario.

    Note:
        - This function assumes a simple linear relationship between stress scenarios
          and portfolio value changes.
        - For more complex portfolios, you might need to implement more sophisticated
          pricing models for each stress scenario.
        - Stress scenarios are represented as percentages. For example:
          - -0.3 represents a 30% decrease in portfolio value
          - 0.2 represents a 20% increase in portfolio value
        - The resulting tensor will have the same shape as the stress_scenarios tensor.
        - Unlike scenario analysis, stress testing typically focuses on adverse scenarios.
    """
    # Calculate the portfolio value under each stress scenario
    # We add 1 to the stress_scenarios to convert percentages to multipliers
    # For example, a -30% scenario becomes a multiplier of 0.7 (1 + (-0.3))
    stressed_values = portfolio_value * (1 + stress_scenarios)

    return stressed_values
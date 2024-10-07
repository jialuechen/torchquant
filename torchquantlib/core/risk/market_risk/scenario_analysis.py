from torch import Tensor

def scenario_analysis(portfolio_value: Tensor, scenarios: Tensor) -> Tensor:
    """
    Perform scenario analysis on a portfolio.

    Scenario analysis is a risk management technique that estimates how portfolio value
    might change under various hypothetical situations or scenarios.

    Args:
        portfolio_value (Tensor): The current value of the portfolio.
        scenarios (Tensor): A tensor of scenario multipliers, where each element represents
                            a different scenario (e.g., market up 10%, down 20%, etc.).

    Returns:
        Tensor: A tensor of portfolio values under each scenario.

    Note:
        - This function assumes a simple linear relationship between scenario changes
          and portfolio value changes.
        - For more complex portfolios, you might need to implement more sophisticated
          pricing models for each scenario.
        - Scenarios are represented as multipliers. For example:
          - 1.1 represents a 10% increase
          - 0.8 represents a 20% decrease
        - The resulting tensor will have the same shape as the scenarios tensor.
    """
    # Calculate the portfolio value under each scenario
    # This is done by multiplying the current portfolio value by each scenario multiplier
    scenario_values = portfolio_value * scenarios

    return scenario_values
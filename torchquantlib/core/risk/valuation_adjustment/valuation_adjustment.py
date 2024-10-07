from torch import Tensor

def calculate_cva(exposure: Tensor, default_prob: Tensor, recovery_rate: Tensor) -> Tensor:
    """
    Calculate the Credit Valuation Adjustment (CVA).

    CVA represents the market value of counterparty credit risk.

    Args:
        exposure (Tensor): Expected positive exposure to the counterparty.
        default_prob (Tensor): Probability of default of the counterparty.
        recovery_rate (Tensor): Expected recovery rate in case of default (between 0 and 1).

    Returns:
        Tensor: The calculated Credit Valuation Adjustment.
    """
    cva = exposure * (1 - recovery_rate) * default_prob
    return cva

def calculate_dva(exposure: Tensor, default_prob: Tensor, recovery_rate: Tensor) -> Tensor:
    """
    Calculate the Debit Valuation Adjustment (DVA).

    DVA is similar to CVA but represents the credit risk of the entity itself.

    Args:
        exposure (Tensor): Expected negative exposure (i.e., liability) to the counterparty.
        default_prob (Tensor): Probability of default of the entity itself.
        recovery_rate (Tensor): Expected recovery rate in case of the entity's default.

    Returns:
        Tensor: The calculated Debit Valuation Adjustment.
    """
    dva = exposure * (1 - recovery_rate) * default_prob
    return dva

def calculate_fva(exposure: Tensor, funding_spread: Tensor, maturity: Tensor) -> Tensor:
    """
    Calculate the Funding Valuation Adjustment (FVA).

    FVA represents the cost of funding for uncollateralized derivatives.

    Args:
        exposure (Tensor): Expected exposure over the life of the derivative.
        funding_spread (Tensor): The entity's funding spread above the risk-free rate.
        maturity (Tensor): Time to maturity of the derivative.

    Returns:
        Tensor: The calculated Funding Valuation Adjustment.
    """
    fva = exposure * funding_spread * maturity
    return fva

def calculate_mva(exposure: Tensor, funding_cost: Tensor, maturity: Tensor) -> Tensor:
    """
    Calculate the Margin Valuation Adjustment (MVA).

    MVA represents the cost of posting initial margin for cleared or non-cleared derivatives.

    Args:
        exposure (Tensor): Expected initial margin requirement.
        funding_cost (Tensor): The cost of funding the initial margin.
        maturity (Tensor): Time to maturity of the derivative.

    Returns:
        Tensor: The calculated Margin Valuation Adjustment.
    """
    mva = exposure * funding_cost * maturity
    return mva
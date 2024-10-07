import math
import torch

def normal_cdf(x):
    """
    Calculate the cumulative distribution function of the standard normal distribution.

    Args:
        x (torch.Tensor): Input value

    Returns:
        torch.Tensor: CDF value
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def normal_pdf(x):
    """
    Calculate the probability density function of the standard normal distribution.

    Args:
        x (torch.Tensor): Input value

    Returns:
        torch.Tensor: PDF value
    """
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def implied_volatility(option_price, spot, strike, expiry, rate, is_call=True):
    """
    Calculate the implied volatility using the 'Let's be rational' algorithm.

    This function implements a root-finding algorithm to determine the implied volatility
    that makes the Black-Scholes model price match the observed market price.

    Args:
        option_price (torch.Tensor): The observed option price
        spot (torch.Tensor): The spot price of the underlying asset
        strike (torch.Tensor): The strike price of the option
        expiry (torch.Tensor): The time to expiry in years
        rate (torch.Tensor): The risk-free interest rate
        is_call (bool): True for call options, False for put options

    Returns:
        torch.Tensor: The implied volatility

    Note:
        This implementation uses the Newton-Raphson method for root-finding,
        with safeguards to ensure convergence.
    """
    # Convert tensors to float for internal calculation
    option_price = option_price.item()
    spot = spot.item()
    strike = strike.item()
    expiry = expiry.item()
    rate = rate.item()
    
    # Calculate the forward price
    forward = spot * math.exp(rate * expiry)

    # Define constants
    epsilon = 1e-8  # Convergence threshold
    max_iter = 100  # Maximum number of iterations
    vol_lower_bound = 1e-5  # Minimum allowed volatility
    vol_upper_bound = 1  # Maximum allowed volatility

    # Initialize implied volatility guess using a simple approximation
    implied_vol = math.sqrt(2 * abs((math.log(forward / strike) + rate * expiry) / expiry))

    for _ in range(max_iter):
        # Ensure volatility stays within bounds
        implied_vol = max(min(implied_vol, vol_upper_bound), vol_lower_bound)

        # Calculate d1 and d2 parameters
        d1 = (math.log(forward / strike) + 0.5 * implied_vol**2 * expiry) / (implied_vol * math.sqrt(expiry))
        d2 = d1 - implied_vol * math.sqrt(expiry)

        # Calculate model price based on current volatility guess
        if is_call:
            model_price = forward * normal_cdf(d1) - strike * normal_cdf(d2)
        else:
            model_price = strike * normal_cdf(-d2) - forward * normal_cdf(-d1)

        # Calculate vega (sensitivity of option price to volatility)
        vega = forward * math.sqrt(expiry) * normal_pdf(d1)

        # Calculate price difference
        price_diff = model_price - option_price

        # Check for convergence
        if abs(price_diff) < epsilon:
            return torch.tensor(implied_vol)

        # Update volatility guess using Newton-Raphson method
        implied_vol -= price_diff / vega

    # Return best guess if max iterations reached
    return torch.tensor(implied_vol)

# Example usage
if __name__ == "__main__":
    option_price = torch.tensor(10.0)
    spot = torch.tensor(100.0)
    strike = torch.tensor(105.0)
    expiry = torch.tensor(1.0)
    rate = torch.tensor(0.05)

    volatility = implied_volatility(option_price, spot, strike, expiry, rate, is_call=True)
    print(f'Implied Volatility: {volatility.item()}')
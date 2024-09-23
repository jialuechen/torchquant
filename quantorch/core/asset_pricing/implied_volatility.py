import math
import torch

def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def normal_pdf(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def implied_volatility(option_price, spot, strike, expiry, rate, is_call=True):
    """
    Calculate the implied volatility using the 'Let's be rational' algorithm.

    Parameters:
    - option_price (torch.Tensor): The observed option price
    - spot (torch.Tensor): The spot price of the underlying asset
    - strike (torch.Tensor): The strike price of the option
    - expiry (torch.Tensor): The time to expiry in years
    - rate (torch.Tensor): The risk-free interest rate
    - is_call (bool): True for call options, False for put options

    Returns:
    - torch.Tensor: The implied volatility
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
    epsilon = 1e-8
    max_iter = 100
    vol_lower_bound = 1e-5
    vol_upper_bound = 1

    # Initialize implied volatility guess
    implied_vol = math.sqrt(2 * abs((math.log(forward / strike) + rate * expiry) / expiry))

    for _ in range(max_iter):
        if implied_vol < vol_lower_bound:
            implied_vol = vol_lower_bound
        if implied_vol > vol_upper_bound:
            implied_vol = vol_upper_bound

        d1 = (math.log(forward / strike) + 0.5 * implied_vol**2 * expiry) / (implied_vol * math.sqrt(expiry))
        d2 = d1 - implied_vol * math.sqrt(expiry)

        if is_call:
            model_price = forward * normal_cdf(d1) - strike * normal_cdf(d2)
        else:
            model_price = strike * normal_cdf(-d2) - forward * normal_cdf(-d1)

        vega = forward * math.sqrt(expiry) * normal_pdf(d1)

        price_diff = model_price - option_price

        if abs(price_diff) < epsilon:
            return torch.tensor(implied_vol)

        implied_vol -= price_diff / vega

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
    
# Example usage:
# iv = implied_volatility('call', 'european', market_price, spot, strike, expiry, rate, dividend)

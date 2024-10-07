from torch import tensor

def zero_coupon_bond(face_value: tensor, rate: tensor, maturity: tensor) -> tensor:
    """
    Calculate the price of a zero-coupon bond.

    Args:
        face_value (tensor): The face value of the bond
        rate (tensor): The interest rate (as a decimal)
        maturity (tensor): Time to maturity in years

    Returns:
        tensor: The price of the zero-coupon bond
    """
    return face_value / (1 + rate)**maturity

def coupon_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor) -> tensor:
    """
    Calculate the price of a coupon-paying bond.

    Args:
        face_value (tensor): The face value of the bond
        coupon_rate (tensor): The coupon rate (as a decimal)
        rate (tensor): The interest rate (as a decimal)
        periods (tensor): Number of coupon periods

    Returns:
        tensor: The price of the coupon bond
    """
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price

def callable_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor, call_price: tensor, call_period: tensor) -> tensor:
    """
    Calculate the price of a callable bond.

    Args:
        face_value (tensor): The face value of the bond
        coupon_rate (tensor): The coupon rate (as a decimal)
        rate (tensor): The interest rate (as a decimal)
        periods (tensor): Number of coupon periods
        call_price (tensor): The price at which the bond can be called
        call_period (tensor): The period at which the bond can be called

    Returns:
        tensor: The price of the callable bond
    """
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        if t == call_period:
            # At call period, use the minimum of regular payment and call price
            bond_price += min(face_value + coupon * t, call_price) / (1 + rate) ** t
        else:
            bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price

def putable_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor, put_price: tensor, put_period:tensor) -> tensor:
    """
    Calculate the price of a putable bond.

    Args:
        face_value (tensor): The face value of the bond
        coupon_rate (tensor): The coupon rate (as a decimal)
        rate (tensor): The interest rate (as a decimal)
        periods (tensor): Number of coupon periods
        put_price (tensor): The price at which the bond can be put
        put_period (tensor): The period at which the bond can be put

    Returns:
        tensor: The price of the putable bond
    """
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        if t == put_period:
            # At put period, use the maximum of regular payment and put price
            bond_price += max(face_value + coupon * t, put_price) / (1 + rate) ** t
        else:
            bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price

def convertible_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor, conversion_ratio: tensor, conversion_price: tensor) -> tensor:
    """
    Calculate the price of a convertible bond.

    Args:
        face_value (tensor): The face value of the bond
        coupon_rate (tensor): The coupon rate (as a decimal)
        rate (tensor): The interest rate (as a decimal)
        periods (tensor): Number of coupon periods
        conversion_ratio (tensor): The number of shares received upon conversion
        conversion_price (tensor): The price of the underlying stock for conversion

    Returns:
        tensor: The price of the convertible bond
    """
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    conversion_value = conversion_ratio * conversion_price
    return min(bond_price, conversion_value)

def stochastic_rate_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor) -> tensor:
    """
    Calculate the price of a bond with stochastic interest rates.

    Args:
        face_value (tensor): The face value of the bond
        coupon_rate (tensor): The coupon rate (as a decimal)
        rate (tensor): A tensor of interest rates for each period
        periods (tensor): Number of coupon periods

    Returns:
        tensor: The price of the bond with stochastic rates
    """
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate[t-1]) ** t
    bond_price += face_value / (1 + rate[-1]) ** periods
    return bond_price
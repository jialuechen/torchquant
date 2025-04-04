import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Optional, Tuple
from scipy.stats import norm
from .black_scholes_merton import black_scholes_merton

ZERO = torch.tensor(0.0)

class ValidationMixin:
    """Mixin class for validating parameters"""
    @staticmethod
    def validate_parameters(volatility: Tensor, expiry: Tensor, num_paths: Optional[int] = None):
        """Validate option parameters"""
        if volatility <= ZERO or expiry <= ZERO:
            raise ValueError("Volatility and expiry must be positive.")
        if num_paths is not None and num_paths <= 0:
            raise ValueError("Number of paths must be a positive integer.")

    @staticmethod
    def validate_option_type(option_type: str):
        """Validate option type"""
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be either 'call' or 'put'.")

class ExoticOption(nn.Module, ValidationMixin):
    """Base class for exotic options"""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

def digital_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor,
                  volatility: Tensor, rate: Tensor, payout: Tensor, num_paths: int) -> Tensor:
    """Price a digital option using Monte Carlo simulation"""
    ValidationMixin.validate_parameters(volatility, expiry, num_paths)
    ValidationMixin.validate_option_type(option_type)

    z = torch.randn(num_paths, device=spot.device)
    asset_prices = spot * torch.exp(
        (rate - 0.5 * volatility**2) * expiry + 
        volatility * torch.sqrt(expiry) * z
    )

    payoffs = ((asset_prices >= strike) if option_type == 'call' else (asset_prices <= strike)).float() * payout
    return (payoffs * torch.exp(-rate * expiry)).mean()

def rainbow_option(spots: List[Tensor], weights: List[Tensor], strike: Tensor, expiry: Tensor,
                  volatilities: List[Tensor], correlations: Tensor, rate: Tensor, num_paths: int) -> Tensor:
    """Price a rainbow option using Monte Carlo simulation"""
    n_assets = len(spots)
    spots = torch.stack(spots)
    vols = torch.stack(volatilities)
    weights = torch.stack(weights)

    # Generate correlated random numbers
    z = torch.randn(num_paths, n_assets, device=spots.device)
    L = torch.linalg.cholesky(correlations)
    corr_rand = z @ L.T

    # Simulate asset prices
    prices = spots.unsqueeze(0) * torch.exp(
        (rate - 0.5 * vols**2).unsqueeze(0) * expiry + 
        vols.unsqueeze(0) * torch.sqrt(expiry) * corr_rand
    )
    
    # Calculate portfolio value
    portfolio = torch.sum(prices * weights, dim=1)
    payoffs = torch.maximum(portfolio - strike, ZERO)
    
    return (payoffs * torch.exp(-rate * expiry)).mean()

def barrier_option(option_type: str, barrier_type: str, spot: Tensor, strike: Tensor, 
                  barrier: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, 
                  num_paths: int, num_steps: int) -> Tensor:    
    """Price a barrier option using Monte Carlo simulation"""
    ValidationMixin.validate_parameters(volatility, expiry, num_paths)
    ValidationMixin.validate_option_type(option_type)
    
    if barrier_type not in ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']:
        raise ValueError("Invalid barrier type")

    dt = expiry / num_steps
    paths = torch.zeros(num_paths, num_steps + 1, device=spot.device)
    paths[:, 0] = spot
    
    # Simulate price paths
    for t in range(1, num_steps + 1):
        z = torch.randn(num_paths, device=spot.device)
        paths[:, t] = paths[:, t-1] * torch.exp(
            (rate - 0.5 * volatility**2) * dt + 
            volatility * torch.sqrt(dt) * z
        )
    
    # Check barrier conditions
    barrier_hit = {
        'up-and-out': torch.any(paths > barrier, dim=1),
        'down-and-out': torch.any(paths < barrier, dim=1),
        'up-and-in': torch.any(paths >= barrier, dim=1),
        'down-and-in': torch.any(paths <= barrier, dim=1)
    }[barrier_type]
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = torch.maximum(paths[:, -1] - strike, ZERO)
    else:
        payoffs = torch.maximum(strike - paths[:, -1], ZERO)
        
    if barrier_type.endswith('out'):
        payoffs = torch.where(barrier_hit, ZERO, payoffs)
    else:
        payoffs = torch.where(barrier_hit, payoffs, ZERO)
    
    return (payoffs * torch.exp(-rate * expiry)).mean()

def lookback_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor,
                   volatility: Tensor, rate: Tensor, num_paths: int, 
                   strike_type: str = 'fixed') -> Tensor:
    """Price a lookback option using Monte Carlo simulation"""
    ValidationMixin.validate_parameters(volatility, expiry, num_paths)
    ValidationMixin.validate_option_type(option_type)
    
    if strike_type not in ['fixed', 'floating']:
        raise ValueError("Invalid strike type")

    dt = expiry / 252  # Daily steps
    num_steps = int(expiry / dt)
    paths = torch.zeros(num_paths, num_steps + 1, device=spot.device)
    paths[:, 0] = spot
    
    # Simulate price paths
    for t in range(1, num_steps + 1):
        z = torch.randn(num_paths, device=spot.device)
        paths[:, t] = paths[:, t-1] * torch.exp(
            (rate - 0.5 * volatility**2) * dt + 
            volatility * torch.sqrt(dt) * z
        )
    
    # Calculate payoffs
    if option_type == 'call':
        if strike_type == 'fixed':
            payoffs = torch.maximum(paths.max(dim=1).values - strike, ZERO)
        else:  # floating
            payoffs = torch.maximum(paths[:, -1] - paths.max(dim=1).values, ZERO)
    else:  # put
        if strike_type == 'fixed':
            payoffs = torch.maximum(strike - paths.min(dim=1).values, ZERO)
        else:  # floating
            payoffs = torch.maximum(paths.min(dim=1).values - paths[:, -1], ZERO)
    
    return (payoffs * torch.exp(-rate * expiry)).mean()

def asian_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor,
                volatility: Tensor, rate: Tensor, num_paths: int, num_steps: int,
                average_type: str = 'arithmetic') -> Tensor:
    """Price an Asian option using Monte Carlo simulation"""
    ValidationMixin.validate_parameters(volatility, expiry, num_paths)
    ValidationMixin.validate_option_type(option_type)
    
    if average_type not in ['arithmetic', 'geometric']:
        raise ValueError("Invalid average type")

    dt = expiry / num_steps
    paths = torch.zeros(num_paths, num_steps + 1, device=spot.device)
    paths[:, 0] = spot
    
    # Simulate price paths
    for t in range(1, num_steps + 1):
        z = torch.randn(num_paths, device=spot.device)
        paths[:, t] = paths[:, t-1] * torch.exp(
            (rate - 0.5 * volatility**2) * dt + 
            volatility * torch.sqrt(dt) * z
        )
    
    # Calculate average price
    if average_type == 'arithmetic':
        avg_price = paths.mean(dim=1)
    else:  # geometric
        avg_price = torch.exp(torch.log(paths).mean(dim=1))
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = torch.maximum(avg_price - strike, ZERO)
    else:  # put
        payoffs = torch.maximum(strike - avg_price, ZERO)
    
    return (payoffs * torch.exp(-rate * expiry)).mean()

def quanto_option(spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor,
                 fx_volatility: Tensor, correlation: Tensor, domestic_rate: Tensor,
                 foreign_rate: Tensor, fx_rate: Tensor, num_paths: int) -> Tensor:
    """Price a quanto option using Monte Carlo simulation"""
    ValidationMixin.validate_parameters(volatility, expiry, num_paths)

    # Generate correlated random numbers
    z1 = torch.randn(num_paths, device=spot.device)
    z2 = correlation * z1 + torch.sqrt(1 - correlation**2) * torch.randn(num_paths, device=spot.device)

    # Simulate asset and FX rate paths
    asset_prices = spot * torch.exp(
        (foreign_rate - 0.5 * volatility**2) * expiry +
        volatility * torch.sqrt(expiry) * z1
    )
    
    fx_rates = fx_rate * torch.exp(
        (domestic_rate - foreign_rate - 0.5 * fx_volatility**2) * expiry +
        fx_volatility * torch.sqrt(expiry) * z2
    )

    # Calculate quanto payoffs
    payoffs = torch.maximum(asset_prices - strike, ZERO) * fx_rates
    return (payoffs * torch.exp(-domestic_rate * expiry)).mean()

def basket_option(spots: List[Tensor], weights: List[Tensor], strike: Tensor, expiry: Tensor,
                 volatilities: List[Tensor], correlations: Tensor, rate: Tensor,
                 option_type: str, num_paths: int) -> Tensor:
    """Price a basket option using Monte Carlo simulation"""
    ValidationMixin.validate_parameters(torch.tensor(volatilities).min(), expiry, num_paths)
    ValidationMixin.validate_option_type(option_type)

    n_assets = len(spots)
    spots = torch.stack(spots)
    vols = torch.stack(volatilities)
    weights = torch.stack(weights)

    # Generate correlated random numbers
    z = torch.randn(num_paths, n_assets, device=spots.device)
    L = torch.linalg.cholesky(correlations)
    corr_rand = z @ L.T

    # Simulate asset prices
    prices = spots.unsqueeze(0) * torch.exp(
        (rate - 0.5 * vols**2).unsqueeze(0) * expiry +
        vols.unsqueeze(0) * torch.sqrt(expiry) * corr_rand
    )

    # Calculate basket value
    basket_value = torch.sum(prices * weights, dim=1)

    # Calculate payoffs
    if option_type == 'call':
        payoffs = torch.maximum(basket_value - strike, ZERO)
    else:  # put
        payoffs = torch.maximum(strike - basket_value, ZERO)

    return (payoffs * torch.exp(-rate * expiry)).mean()

class CurranAsianOption(ExoticOption):
    """Price an Asian option using Curran's approximation"""
    def __init__(self, option_type: str, average_type: str = 'arithmetic'):
        super().__init__()
        self.option_type = option_type
        self.average_type = average_type
        self.validate_option_type(option_type)
        
    def forward(self, spot: Tensor, strike: Tensor, expiry: Tensor,
               volatility: Tensor, rate: Tensor, num_steps: int) -> Tensor:
        """Calculate the price using Curran's approximation"""
        self.validate_parameters(volatility, expiry)
        
        dt = expiry / num_steps
        adjusted_vol = volatility * torch.sqrt((num_steps + 1) / (6 * num_steps))
        
        if self.average_type == 'arithmetic':
            # Curran approximation
            d1 = (torch.log(spot/strike) + (rate + 0.5 * adjusted_vol**2) * expiry) / (adjusted_vol * torch.sqrt(expiry))
            d2 = d1 - adjusted_vol * torch.sqrt(expiry)
            
            if self.option_type == 'call':
                price = spot * torch.exp(-rate * expiry) * norm.cdf(d1) - strike * torch.exp(-rate * expiry) * norm.cdf(d2)
            else:
                price = strike * torch.exp(-rate * expiry) * norm.cdf(-d2) - spot * torch.exp(-rate * expiry) * norm.cdf(-d1)
        else:
            # Analytical solution for geometric average
            adjusted_rate = rate - 0.5 * (volatility**2) / 3
            price = black_scholes_merton(
                self.option_type, 'european', spot, strike, expiry,
                adjusted_vol, adjusted_rate, torch.tensor(0.0)
            )
            
        return price

def chooser_option(spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor,
                  rate: Tensor, choose_time: Tensor) -> Tensor:
    """Price a chooser option using Black-Scholes formula"""
    ValidationMixin.validate_parameters(volatility, expiry)

    # Calculate call and put prices at the choose time
    d1 = (torch.log(spot/strike) + (rate + 0.5 * volatility**2) * (expiry - choose_time)) / (volatility * torch.sqrt(expiry - choose_time))
    d2 = d1 - volatility * torch.sqrt(expiry - choose_time)

    call_price = spot * norm.cdf(d1) - strike * torch.exp(-rate * (expiry - choose_time)) * norm.cdf(d2)
    put_price = strike * torch.exp(-rate * (expiry - choose_time)) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return torch.maximum(call_price, put_price) * torch.exp(-rate * choose_time)

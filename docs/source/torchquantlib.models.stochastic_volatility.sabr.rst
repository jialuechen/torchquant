SABR Model
==========

.. automodule:: torchquantlib.models.stochastic_volatility.sabr

.. autoclass:: SABR
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: simulate

   .. automethod:: option_price

   .. automethod:: _apply_constraints

Class Methods
-------------

__init__(alpha_init=0.3, beta_init=0.5, rho_init=-0.5, nu_init=0.4, F0=100.0)
    Initialize the SABR model.

    :param float alpha_init: Initial value for volatility.
    :param float beta_init: Initial value for elasticity parameter (0 ≤ β ≤ 1).
    :param float rho_init: Initial value for correlation between F and α processes.
    :param float nu_init: Initial value for volatility of volatility.
    :param float F0: Initial forward rate.

simulate(S0, T, N, steps=100)
    Simulate forward rate paths using the SABR model.

    :param float S0: Initial asset price (not used, included for consistency with other models).
    :param float T: Time horizon for simulation.
    :param int N: Number of simulation paths.
    :param int steps: Number of time steps in each path.
    :return: Simulated forward rates at time T.
    :rtype: torch.Tensor

option_price(K, T, r, option_type='call', N=10000, steps=100)
    Calculate the option price using Monte Carlo simulation.

    :param float K: Strike price.
    :param float T: Time to maturity.
    :param float r: Risk-free interest rate.
    :param str option_type: 'call' or 'put'.
    :param int N: Number of simulation paths.
    :param int steps: Number of time steps in simulation.
    :return: Option price.
    :rtype: float

_apply_constraints()
    Apply constraints to model parameters to ensure they remain in valid ranges.

Model Description
-----------------

The SABR (Stochastic Alpha, Beta, Rho) model is a stochastic volatility model used in mathematical finance to model forward rates and option prices. It is particularly useful for modeling the volatility smile in interest rate derivatives.

The model is described by two stochastic differential equations:

.. math::

   dF_t &= \alpha_t F_t^\beta dW^1_t \\
   d\alpha_t &= \nu \alpha_t dW^2_t

where:

- :math:`F_t` is the forward rate at time t
- :math:`\alpha_t` is the stochastic volatility at time t
- :math:`\beta` is the elasticity parameter (0 ≤ β ≤ 1)
- :math:`\nu` is the volatility of volatility
- :math:`\rho` is the correlation between :math:`W^1_t` and :math:`W^2_t` (the two Wiener processes)
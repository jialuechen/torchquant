Heston Model
============

.. automodule:: torchquantlib.models.stochastic_volatility.heston

.. autoclass:: Heston
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: simulate

   .. automethod:: _apply_constraints

Class Methods
-------------

__init__(kappa_init=2.0, theta_init=0.04, sigma_init=0.3, rho_init=-0.7, v0_init=0.04)
    Initialize the Heston model.

    :param float kappa_init: Initial value for mean reversion speed of variance.
    :param float theta_init: Initial value for long-term mean of variance.
    :param float sigma_init: Initial value for volatility of variance.
    :param float rho_init: Initial value for correlation between asset returns and variance.
    :param float v0_init: Initial variance.

simulate(S0, T, N, steps=100)
    Simulate asset price paths using the Heston model.

    :param float S0: Initial asset price.
    :param float T: Time horizon for simulation.
    :param int N: Number of simulation paths.
    :param int steps: Number of time steps in each path.
    :return: Simulated asset prices at time T.
    :rtype: torch.Tensor

_apply_constraints()
    Apply constraints to model parameters to ensure they remain in valid ranges.

Model Description
-----------------

The Heston model is a stochastic volatility model used in mathematical finance to model the evolution of asset prices and their volatility. It is particularly useful for pricing options and modeling the volatility smile observed in financial markets.

The model is described by two stochastic differential equations:

.. math::

   dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW^1_t \\
   dv_t &= \kappa(\theta - v_t) dt + \sigma \sqrt{v_t} dW^2_t

where:

- :math:`S_t` is the asset price at time t
- :math:`v_t` is the instantaneous variance at time t
- :math:`\mu` is the drift of the asset price (often set to the risk-free rate)
- :math:`\kappa` is the rate of mean reversion of the variance
- :math:`\theta` is the long-term mean of the variance
- :math:`\sigma` is the volatility of volatility
- :math:`\rho` is the correlation between :math:`W^1_t` and :math:`W^2_t` (the two Wiener processes)

The Heston model allows for a more flexible and realistic representation of asset price dynamics compared to models with constant volatility, as it captures the stochastic nature of volatility and can reproduce observed market phenomena such as volatility clustering and leverage effects.
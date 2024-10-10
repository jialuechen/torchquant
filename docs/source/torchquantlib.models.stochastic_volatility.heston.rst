Heston Model
============

.. currentmodule:: torchquantlib.models.stochastic_volatility.heston

.. autoclass:: Heston
   :members:
   :undoc-members:
   :show-inheritance:

   The Heston model is a stochastic volatility model used in financial mathematics to model the evolution of asset prices and their volatility. It is particularly useful for pricing options and other derivatives.

   The model is described by two stochastic differential equations:

   .. math::

      dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW^S_t \\
      dv_t &= \kappa(\theta - v_t)dt + \sigma_v \sqrt{v_t} dW^v_t

   where:

   - :math:`S_t` is the asset price at time t
   - :math:`v_t` is the variance at time t
   - :math:`\mu` is the drift of the asset price
   - :math:`\kappa` is the rate of mean reversion of variance
   - :math:`\theta` is the long-term mean of variance
   - :math:`\sigma_v` is the volatility of variance
   - :math:`\rho` is the correlation between the two Wiener processes :math:`W^S_t` and :math:`W^v_t`

   .. automethod:: __init__

      :param float kappa_init: Initial value for rate of mean reversion of variance.
      :param float theta_init: Initial value for long-term mean of variance.
      :param float sigma_v_init: Initial value for volatility of variance.
      :param float rho_init: Initial value for correlation between asset and variance processes.
      :param float v0_init: Initial value for variance.
      :param float mu_init: Initial value for drift of asset price.

   .. automethod:: simulate

      :param float S0: Initial asset price.
      :param float T: Time horizon for simulation.
      :param int N: Number of simulation paths.
      :param int steps: Number of time steps in each path.
      :return: Simulated asset prices at time T.
      :rtype: torch.Tensor

   .. automethod:: option_price

      :param float S0: Initial asset price.
      :param float K: Strike price.
      :param float T: Time to maturity.
      :param float r: Risk-free interest rate.
      :param str option_type: 'call' or 'put'.
      :param int N: Number of simulation paths.
      :param int steps: Number of time steps in simulation.
      :return: Option price.
      :rtype: float

   .. automethod:: _apply_constraints

      This method applies constraints to model parameters to ensure they remain in valid ranges.
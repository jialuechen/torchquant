Vasicek Model
==============

.. automodule:: torchquantlib.models.interest_rate.vasicek

.. autoclass:: Vasicek
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: simulate

   .. automethod:: _apply_constraints

Class Methods
-------------

__init__(kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03)
    Initialize the Vasicek model.

    :param float kappa_init: Initial value for speed of mean reversion.
    :param float theta_init: Initial value for long-term mean level.
    :param float sigma_init: Initial value for volatility.
    :param float r0_init: Initial interest rate.

simulate(S0, T, N, steps=100)
    Simulate interest rate paths using the Vasicek model.

    :param float S0: Initial asset price (not used in this model, included for consistency).
    :param float T: Time horizon for simulation.
    :param int N: Number of simulation paths.
    :param int steps: Number of time steps in each path.
    :return: Simulated interest rates at time T.
    :rtype: torch.Tensor

_apply_constraints()
    Apply constraints to model parameters to ensure they remain in valid ranges.
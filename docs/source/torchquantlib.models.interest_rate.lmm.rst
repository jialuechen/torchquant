Libor Market Model (LMM)
=========================

.. automodule:: torchquantlib.models.interest_rate.lmm

.. autoclass:: LMM
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: simulate

   .. automethod:: _apply_constraints

Class Methods
-------------

__init__(forward_rates_init, volatilities_init, correlations_init)
    Initialize the Libor Market Model.

    :param list forward_rates_init: Initial forward rates.
    :param list volatilities_init: Initial volatilities for each forward rate.
    :param numpy.ndarray correlations_init: Correlation matrix for the forward rates.

simulate(S0, T, N, steps=100)
    Simulate forward rate paths using the Libor Market Model.

    :param float S0: Initial asset price (not used in this model, included for consistency).
    :param float T: Time horizon for simulation.
    :param int N: Number of simulation paths.
    :param int steps: Number of time steps in each path.
    :return: Simulated forward rates at time T for all tenors.
    :rtype: torch.Tensor

_apply_constraints()
    Apply constraints to model parameters to ensure they remain in valid ranges.
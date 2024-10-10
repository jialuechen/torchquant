Malliavin Greeks
================

.. automodule:: torchquantlib.core.risk.greeks.greeks
   :members:
   :undoc-members:
   :show-inheritance:


.. autoclass:: torchquantlib.core.risk.greeks.greeks.MalliavinGreeks
   :members:
   :undoc-members:
   :special-members: __init__

   .. automethod:: european_option_greeks

   Usage Example:

   .. code-block:: python

      from torchquantlib.core.risk.greeks.greeks import MalliavinGreeks

      malliavin = MalliavinGreeks()
      greeks = malliavin.european_option_greeks(S0=100, K=100, T=1, r=0.05, sigma=0.2)
      print(greeks)

   .. automethod:: digital_option_greeks

   Usage Example:

   .. code-block:: python

      greeks = malliavin.digital_option_greeks(S0=100, K=100, T=1, r=0.05, sigma=0.2, Q=10)
      print(greeks)

   .. automethod:: barrier_option_greeks

   Usage Example:

   .. code-block:: python

      greeks = malliavin.barrier_option_greeks(S0=100, K=100, H=120, T=1, r=0.05, sigma=0.2, option_type='up-and-out')
      print(greeks)

   .. automethod:: lookback_option_delta

   Usage Example:

   .. code-block:: python

      delta = malliavin.lookback_option_delta(S0=100, T=1, r=0.05, sigma=0.2)
      print(delta)

   .. automethod:: basket_option_greeks

   Usage Example:

   .. code-block:: python

      S0 = [100, 110]
      sigma = [0.2, 0.25]
      rho = [[1, 0.5], [0.5, 1]]
      weights = [0.6, 0.4]
      greeks = malliavin.basket_option_greeks(S0=S0, K=105, T=1, r=0.05, sigma=sigma, rho=rho, weights=weights)
      print(greeks)

   .. automethod:: asian_option_greeks

   Usage Example:

   .. code-block:: python

      greeks = malliavin.asian_option_greeks(S0=100, K=100, T=1, r=0.05, sigma=0.2)
      print(greeks)

This class provides methods to calculate Greeks for various types of options using Malliavin calculus. The methods use Monte Carlo simulation and are particularly useful for complex options where closed-form solutions are not available.

Note:
   - All methods use PyTorch tensors for calculations and can leverage GPU acceleration if available.
   - The number of Monte Carlo paths and other simulation parameters can be adjusted for a trade-off between accuracy and computation time.
   - For some complex options, only specific Greeks are calculated due to the complexity of the calculations.


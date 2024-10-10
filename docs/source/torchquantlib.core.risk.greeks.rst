Greeks
======

This class provides methods to calculate Greeks for various types of options using Malliavin calculus. The methods use Monte Carlo simulation and are particularly useful for complex options where closed-form solutions are not available.


.. automodule:: torchquantlib.core.risk.greeks
   

.. autoclass:: Greeks
   :members:
   :undoc-members:
   :show-inheritance:

Malliavin Greeks for european_option_greeks
------------

   .. code-block:: python

      from torchquantlib.core.risk.greeks.greeks import Greeks

      greeks = Greeks()
      results = greeks.european_option_greeks(S0=100, K=100, T=1, r=0.05, sigma=0.2)
      print(results)

   .. automethod:: digital_option_greeks

Malliavin Greeks for Digital Options
------------

   .. code-block:: python

      results = greeks.digital_option_greeks(S0=100, K=100, T=1, r=0.05, sigma=0.2, Q=10)
      print(results)

Malliavin Greeks for Barrier Options
------------

   .. code-block:: python

      results = greeks.barrier_option_greeks(S0=100, K=100, H=120, T=1, r=0.05, sigma=0.2, option_type='up-and-out')
      print(results)

   .. automethod:: lookback_option_delta

Malliavin Greeks for Lookback Options
------------

   .. code-block:: python

      delta = greeks.lookback_option_delta(S0=100, T=1, r=0.05, sigma=0.2)
      print(delta)

   .. automethod:: basket_option_greeks

Malliavin Greeks for Basket Options
------------

   .. code-block:: python

      S0 = [100, 110]
      sigma = [0.2, 0.25]
      rho = [[1, 0.5], [0.5, 1]]
      weights = [0.6, 0.4]
      results = greeks.basket_option_greeks(S0=S0, K=105, T=1, r=0.05, sigma=sigma, rho=rho, weights=weights)
      print(results)

   .. automethod:: asian_option_greeks

Malliavin Greeks for Asian Options:
------------

   .. code-block:: python

      results = greeks.asian_option_greeks(S0=100, K=100, T=1, r=0.05, sigma=0.2)
      print(results)


Note:
   - All methods use PyTorch tensors for calculations and can leverage GPU acceleration if available.
   - The number of Monte Carlo paths and other simulation parameters can be adjusted for a trade-off between accuracy and computation time.
   - For some complex options, only specific Greeks are calculated due to the complexity of the calculations.

Dupire Local Volatility Model
=============================

.. currentmodule:: torchquantlib.models.local_volatility.dupire_local_volatility

.. autoclass:: DupireLocalVol
   :show-inheritance:
   :members:
   :special-members: __init__

   The Dupire Local Volatility model extends the Black-Scholes model by allowing the volatility to be a function
   of both the underlying asset price and time. It is based on Dupire's formula, which relates the local volatility
   to the implied volatility surface.

   The model is described by the following stochastic differential equation:

   .. math::

      dS = \mu S dt + \sigma(S,t) S dW

   where:

   - :math:`S` is the asset price
   - :math:`\mu` is the drift (usually the risk-free rate)
   - :math:`\sigma(S,t)` is the local volatility function
   - :math:`W` is a Wiener process

   .. rubric:: Methods

   .. automethod:: __init__
   .. automethod:: simulate

   .. rubric:: Attributes

   .. attribute:: local_vol_func
      :type: callable

      A function that takes :math:`(S, t)` and returns local volatility :math:`\sigma(S, t)`.
      :math:`S` can be a tensor of asset prices, and :math:`t` is a scalar time value.

   .. rubric:: Example Usage

   .. code-block:: python

      import torch
      from torchquantlib.models.local_volatility.dupire_local_volatility import DupireLocalVol

      # Define a simple local volatility function
      def local_vol_func(S, t):
          return 0.2 + 0.1 * torch.exp(-S / 100) + 0.05 * t

      # Initialize the Dupire Local Volatility model
      model = DupireLocalVol(local_vol_func)

      # Simulate asset price paths
      S0 = 100.0  # Initial asset price
      T = 1.0     # Time horizon
      N = 10000   # Number of simulation paths
      steps = 252 # Number of time steps (e.g., daily steps for a year)

      simulated_prices = model.simulate(S0, T, N, steps)

   .. note::
      The `simulate` method returns the final asset prices at time `T`. If you need the entire price path,
      you can modify the method to return the full `S` tensor.

   .. seealso::
      - :class:`torchquantlib.models.stochastic_model.StochasticModel`
      - :doc:`Black-Scholes Model </torchquantlib.models.black_scholes>`
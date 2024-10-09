American Option
===============

.. automodule:: torchquantlib.core.asset_pricing.option.american_option
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The American Option module provides functionality for pricing and analyzing American-style options using various numerical methods. American options can be exercised at any time up to the expiration date, making them more complex to value than European options.

Key Features
------------

- Pricing of American call and put options
- Support for various underlying assets (e.g., stocks, indices)
- Implementation of multiple pricing methods (e.g., binomial tree, finite difference)
- Greeks calculation for risk management

Classes
-------

AmericanOption
~~~~~~~~~~~~~~

.. autoclass:: torchquantlib.core.asset_pricing.option.american_option.AmericanOption
   :members:
   :undoc-members:
   :show-inheritance:

   The AmericanOption class represents an American-style option contract and provides methods for pricing and analyzing the option.

   :param underlying: The price of the underlying asset
   :type underlying: float
   :param strike: The strike price of the option
   :type strike: float
   :param expiry: Time to expiration in years
   :type expiry: float
   :param rf_rate: Risk-free interest rate (annualized)
   :type rf_rate: float
   :param volatility: Volatility of the underlying asset (annualized)
   :type volatility: float
   :param option_type: Type of the option ('call' or 'put')
   :type option_type: str

   .. method:: price(method='binomial', steps=100)

      Calculate the price of the American option using the specified method.

      :param method: Pricing method to use ('binomial' or 'finite_difference')
      :type method: str
      :param steps: Number of steps in the pricing model
      :type steps: int
      :return: The calculated option price
      :rtype: float

   .. method:: delta()

      Calculate the delta of the American option.

      :return: The option's delta
      :rtype: float

   .. method:: gamma()

      Calculate the gamma of the American option.

      :return: The option's gamma
      :rtype: float

   .. method:: theta()

      Calculate the theta of the American option.

      :return: The option's theta
      :rtype: float

   .. method:: vega()

      Calculate the vega of the American option.

      :return: The option's vega
      :rtype: float

   .. method:: rho()

      Calculate the rho of the American option.

      :return: The option's rho
      :rtype: float

Functions
---------

.. autofunction:: torchquantlib.core.asset_pricing.option.american_option.binomial_tree_price

   Implement the binomial tree method for pricing American options.

   :param option: An instance of the AmericanOption class
   :type option: AmericanOption
   :param steps: Number of steps in the binomial tree
   :type steps: int
   :return: The calculated option price
   :rtype: float

.. autofunction:: torchquantlib.core.asset_pricing.option.american_option.finite_difference_price

   Implement the finite difference method for pricing American options.

   :param option: An instance of the AmericanOption class
   :type option: AmericanOption
   :param steps: Number of steps in the finite difference grid
   :type steps: int
   :return: The calculated option price
   :rtype: float

Examples
--------

Here's a basic example of how to use the AmericanOption class:

.. code-block:: python

   from torchquantlib.core.asset_pricing.option.american_option import AmericanOption

   # Create an American call option
   option = AmericanOption(
       underlying=100,
       strike=95,
       expiry=1.0,
       rf_rate=0.05,
       volatility=0.2,
       option_type='call'
   )

   # Price the option using the binomial tree method
   price = option.price(method='binomial', steps=1000)
   print(f"American call option price: {price:.4f}")

   # Calculate option Greeks
   delta = option.delta()
   gamma = option.gamma()
   theta = option.theta()
   vega = option.vega()
   rho = option.rho()

   print(f"Delta: {delta:.4f}")
   print(f"Gamma: {gamma:.4f}")
   print(f"Theta: {theta:.4f}")
   print(f"Vega: {vega:.4f}")
   print(f"Rho: {rho:.4f}")
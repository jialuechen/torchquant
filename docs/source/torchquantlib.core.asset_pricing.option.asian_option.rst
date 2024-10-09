Asian Option
============

.. automodule:: torchquantlib.core.asset_pricing.option.asian_option
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Asian Option module provides functionality for pricing and analyzing Asian-style options using various numerical methods. Asian options are path-dependent options where the payoff depends on the average price of the underlying asset over a specific period.

Key Features
------------

- Pricing of Asian call and put options
- Support for arithmetic and geometric average price calculations
- Implementation of multiple pricing methods (e.g., Monte Carlo simulation, analytical approximations)
- Greeks calculation for risk management

Classes
-------

AsianOption
~~~~~~~~~~~

.. autoclass:: torchquantlib.core.asset_pricing.option.asian_option.AsianOption
   :members:
   :undoc-members:
   :show-inheritance:

   The AsianOption class represents an Asian-style option contract and provides methods for pricing and analyzing the option.

   :param underlying: The initial price of the underlying asset
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
   :param averaging_type: Type of averaging ('arithmetic' or 'geometric')
   :type averaging_type: str
   :param averaging_period: Period over which the average is calculated (in years)
   :type averaging_period: float

   .. method:: price(method='monte_carlo', num_simulations=10000)

      Calculate the price of the Asian option using the specified method.

      :param method: Pricing method to use ('monte_carlo' or 'analytical_approximation')
      :type method: str
      :param num_simulations: Number of simulations for Monte Carlo method
      :type num_simulations: int
      :return: The calculated option price
      :rtype: float

   .. method:: delta()

      Calculate the delta of the Asian option.

      :return: The option's delta
      :rtype: float

   .. method:: gamma()

      Calculate the gamma of the Asian option.

      :return: The option's gamma
      :rtype: float

   .. method:: theta()

      Calculate the theta of the Asian option.

      :return: The option's theta
      :rtype: float

   .. method:: vega()

      Calculate the vega of the Asian option.

      :return: The option's vega
      :rtype: float

   .. method:: rho()

      Calculate the rho of the Asian option.

      :return: The option's rho
      :rtype: float

Functions
---------

.. autofunction:: torchquantlib.core.asset_pricing.option.asian_option.monte_carlo_price

   Implement the Monte Carlo simulation method for pricing Asian options.

   :param option: An instance of the AsianOption class
   :type option: AsianOption
   :param num_simulations: Number of Monte Carlo simulations
   :type num_simulations: int
   :return: The calculated option price
   :rtype: float

.. autofunction:: torchquantlib.core.asset_pricing.option.asian_option.analytical_approximation_price

   Implement an analytical approximation method for pricing Asian options.

   :param option: An instance of the AsianOption class
   :type option: AsianOption
   :return: The calculated option price
   :rtype: float

Examples
--------

Here's a basic example of how to use the AsianOption class:

.. code-block:: python

   from torchquantlib.core.asset_pricing.option.asian_option import AsianOption

   # Create an Asian call option
   option = AsianOption(
       underlying=100,
       strike=95,
       expiry=1.0,
       rf_rate=0.05,
       volatility=0.2,
       option_type='call',
       averaging_type='arithmetic',
       averaging_period=0.5
   )

   # Price the option using the Monte Carlo method
   price = option.price(method='monte_carlo', num_simulations=100000)
   print(f"Asian call option price: {price:.4f}")

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
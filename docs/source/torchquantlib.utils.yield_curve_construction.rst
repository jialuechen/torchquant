Yield Curve Construction
========================

.. automodule:: torchquantlib.utils.yield_curve_construction
   :members:
   :undoc-members:
   :show-inheritance:

Bootstrap Yield Curve
---------------------

.. autofunction:: torchquantlib.utils.yield_curve_construction.bootstrap_yield_curve

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.utils.yield_curve_construction import bootstrap_yield_curve

   # Example cash flows for 3 instruments with different maturities
   cash_flows = torch.tensor([
       [100, 0, 0],
       [0, 100, 0],
       [0, 0, 100]
   ])

   # Corresponding market prices
   prices = torch.tensor([98, 95, 90])

   # Calculate yields
   yields = bootstrap_yield_curve(cash_flows, prices)
   print(yields)

Nelson-Siegel Yield Curve
-------------------------

.. autofunction:: torchquantlib.utils.yield_curve_construction.nelson_siegel_yield_curve

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchquantlib.utils.yield_curve_construction import nelson_siegel_yield_curve

   # Time to maturities
   tau = torch.tensor([0.5, 1, 2, 3, 5, 10])

   # Nelson-Siegel parameters
   beta0 = torch.tensor(0.03)
   beta1 = torch.tensor(-0.02)
   beta2 = torch.tensor(0.01)

   # Calculate yields
   yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
   print(yields)

Examples
========

This section provides examples of how to use TorchQuant for various quantitative finance tasks.

Model Calibration
-----------------

Here's an example of calibrating a model using TorchQuant:

.. code-block:: python

   import torch
   import torchquant as tq

   # Load market data
   market_data = tq.utils.load_market_data('path/to/data.csv')

   # Create a model
   model = tq.models.HestonModel()

   # Create a calibrator
   calibrator = tq.calibration.ModelCalibrator(model, market_data)

   # Perform calibration
   calibrated_params = calibrator.calibrate()

   print("Calibrated parameters:", calibrated_params)

More examples covering different aspects of TorchQuant will be added in future updates.
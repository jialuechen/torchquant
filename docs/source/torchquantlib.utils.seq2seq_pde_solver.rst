torchquantlib.utils.seq2seq_pde_solver
==================

.. automodule:: torchquantlib.utils.seq2seq_pde_solver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: torchquantlib.utils.seq2seq_pde_solver.Seq2SeqPDESolver
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. rubric:: Methods

   .. automethod:: forward
   .. automethod:: train_step
   .. automethod:: validate_step
   .. automethod:: configure_optimizers

   .. rubric:: Attributes

   .. autoattribute:: encoder
   .. autoattribute:: decoder
   .. autoattribute:: criterion

   .. rubric:: Example Usage

   .. code-block:: python

      import torch
      from torchquantlib.utils.seq2seq_pde_solver import Seq2SeqPDESolver

      # Define your encoder and decoder architectures
      encoder = YourEncoderClass(...)
      decoder = YourDecoderClass(...)

      # Initialize the Seq2SeqPDESolver
      pde_solver = Seq2SeqPDESolver(encoder, decoder)

      # Prepare your input data
      input_data = torch.randn(batch_size, sequence_length, input_dim)

      # Solve the PDE
      output = pde_solver(input_data)

   .. note::
      Make sure to replace `YourEncoderClass` and `YourDecoderClass` with your actual encoder and decoder implementations.

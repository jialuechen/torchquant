Installation
============

This section covers the installation process for TorchQuant.

Prerequisites
-------------

Before installing TorchQuant, ensure you have the following:

- Python 3.7 or higher
- pip (Python package installer)

Installing from PyPI
--------------------

The easiest way to install TorchQuant is using pip:

.. code-block:: bash

    pip install torchquantlib

Installing from Source
----------------------

To install TorchQuant from source:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/jialuechen/torchquant.git
       cd torchquant

2. Install the package:

   .. code-block:: bash

       pip install .

Verifying the Installation
--------------------------

To verify that TorchQuant has been installed correctly, you can run:

.. code-block:: python

    import torchquantlib
    print(torchquantlib.__version__)

This should print the version number of the installed package.

Upgrading
---------

To upgrade to the latest version of TorchQuant, use:

.. code-block:: bash

    pip install --upgrade torchquantlib

Troubleshooting
---------------

If you encounter any issues during installation, please check our :doc:`troubleshooting` guide or open an issue on our GitHub repository.
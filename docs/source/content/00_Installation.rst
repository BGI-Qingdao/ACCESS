.. _installation-guide:

Installation
==================


Our tools should be installed on a Linux system with Python3.8+.

Installing with a 'package manager'
----------------------------------------

System Requirements
++++++++++++++++++++++++++++++++
- **Hardware Configuration**:

  - NVIDIA GPU (VRAM ≥48GB, 16 - core CPU recommended)
- **Software Environment**:

  - CUDA 12.4+ driver
  - Python 3.8+ interpreter
  - Conda package management tool

We strongly recommend your installation executed in an isolated conda environment, so firstly run:

Quick Installation Process
+++++++++++++++++++++++++++++++

1. Install the Rosetta Energy Calculation Suite
****************************************************
.. code-block:: bash

   cd rosetta
   
   # Download the Rosetta suite
   wget https://downloads.rosettacommons.org/downloads/academic/3.14/rosetta_bin_linux_3.14_bundle.tar.bz2
   
   # Extract the package
   tar -xvf rosetta_bin_linux_3.14_bundle.tar.bz2 

   # Go back to the main directory
   cd ..

2. Configure the Python Virtual Environment
*******************************************************
.. code-block:: bash

   # Create an isolated environment and activate it
   conda create -n ACCESS python=3.8
   conda activate ACCESS
   
   # Install the PyTorch core framework
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

3. Install Core Dependencies
*******************************************************
.. code-block:: bash

   pip install argparse biopython torchdrug torch-scatter fair-esm scipy natsort

4. Compile the Geometric Vector Perceptor (GVP) Library
******************************************************************
.. code-block:: bash

   # Clone the official repository and apply the patch
   git clone https://github.com/drorlab/gvp-pytorch.git

   # Apply the patch
   cd gvp-pytorch
   git apply gvp_topology.patch
   pip install .
   cd ..



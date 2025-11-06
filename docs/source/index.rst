.. ACCESS documentation master file, created by
   sphinx-quickstart on Thu Jun  5 16:22:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. =====================
.. Document Title
.. =====================

.. First level
.. -----------

.. Second level
.. ++++++++++++

.. Third level
.. ************

.. Fourth level
.. ~~~~~~~~~~~~


ACCESS
========================================================
.. rubric:: Decoding the Energetic Blueprint of Extremophile Enzyme: A Multimodal AI Framework for Functional Discovery Beyond Sequence and Structure Homology.

**ACCESS** is a specialized open-souhierarchyrce Python package in the field of biomanufacturing. It aims to offer high-precision EC number prediction for the rational design of industrial enzymes, 
screening of functional proteins, and enzyme reaction optimization using MIT license. By combining 3D protein structural features, side-chain features, and residue-level Rosetta energy, 
it uses a hybrid graph neural network architecture. Based on hierarchical contrastive learning and multi-label adaptive fine-tuning, it predicts protein functions. 
Through a topology-aware gradient attention mechanism, it precisely locates function - critical residues. 
This approach pioneers a multimodal feature fusion graph neural network architecture and an interpretable AI algorithm (“Activity Existence - EC Prediction - Rational Design”), 
surmounting the limitations of conventional tools.

* Get quickly started by browsing :ref:`installation-guide`, :ref:`basic-usage` or :ref:`tutorials-index`.
* Open to discuss and provide feedback on `Github <https://github.com/BGI-Qingdao/ACCESS.git>`_.


Highlights
------------

Conceptual Advances:
++++++++++++++++++++++++

(1) **A multimodal heterogeneous data fusion framework for protein function prediction:** Breaking through the limitations of traditional single - data - modality approaches, 
this framework integrates three types of heterogeneous data - protein sequences, structures, and energies. It establishes a cross - modal feature fusion theoretical system, 
offering a more comprehensive and accurate multi - dimensional information base for protein function prediction.

(2) **A hierarchical contrastive learning - driven model optimization paradigm:** This paradigm incorporates a hierarchical contrastive learning strategy. 
It thoroughly explores and utilizes data features and semantic information at different EC levels. Through multi - level contrastive learning tasks, 
it guides the model to more effectively learn complex protein feature representations, thereby enhancing the model's performance and generalization ability.

Technological Innovation:
++++++++++++++++++++++++++++++
(1) **A hybrid GNN - based EC label function prediction model architecture:** This architecture combines the advantages of various graph neural networks (GNNs), 
such as GVP and GAT. It can better handle protein graph - structured data and capture the complex relationships and features of protein molecules, 
improving the accuracy of EC label function prediction.

(2) **A dual - gradient collaborative dynamic threshold decision - making algorithm:** This innovative algorithm improves the accuracy of multi - label prediction. 
The absolute gradient detection measures the fluctuation of actual changes, while the relative gradient identifies the proportion of mutations relative to the current distance. 
This approach provides more reliable multi - label prediction for the model, ensuring the accuracy and stability of prediction results.

(3) **Topology - aware gradient attention residue localization technology:** Leveraging a topology - aware gradient attention mechanism, 
this technology focuses on the topological relationships and interactions of key residues in protein structures. It accurately locates key residues related to functions, 
offering deeper molecular - level insights for protein function research.

Application Innovation:
++++++++++++++++++++++++++++++

(1) **A block - based parallel processing method for graph - structured data:** This method divides large - scale protein graph - structured data into blocks for storage and loading. 
Along with an index mapping mechanism, it enables streaming block - based loading, significantly optimizing memory usage and computational efficiency. 
To address the frequent I/O access issues caused by traditional random sampling strategies during cross - file operations, a non - cross - file random sampling scheme is adopted. 
This greatly reduces I/O operation time when loading different data blocks, boosting overall data processing performance.

(2) **Expansion and error - correction of the EC system:** The existing EC (Enzyme Commission) system is expanded and refined. By comparing model prediction results with known EC numbers, 
potential errors are identified and corrected, enhancing the accuracy and completeness of the EC system.

(3) **Construction of the MEER functional database:** The MEER functional database is created to integrate data from various sources and prediction results. 
It provides researchers with a comprehensive and systematic resource for protein function information, advancing protein function research and its applications.

MEER Database: Panoramic View of EC Distribution, Energy Features, and Model Architecture
-------------------------------------------------------------------------------------------

.. image:: ./_static/model_architecture_diagram.png
    :alt: Title figure
    :width: 700px
    :align: center


.. toctree::
    :titlesonly:
    :maxdepth: 4
    :hidden:

    content/00_Installation
    content/01_Basic_Usage
    Tutorials/index
    content/03_References
    


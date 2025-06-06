.. _tutorials-index:

Tutorials
================

.. toctree::
    :maxdepth: 1

    train.md
    predict.md

Example Label File
--------------------------
The label file is a CSV that links protein names to their labels. Proteins with no activity are denoted as 0.-.-.-. See the example below for details.

.. csv-table::
   :header: "protein_name", "labels"
   :widths: 50, 50

   "AF-A0A009IHW8-F1-model_v4_0001", "3.2.2.6"
   "AF-A0A024RXP8-F1-model_v4_0001", "3.2.1.91"
   "AF-A0A024SH76-F1-model_v4_0001", "3.2.1.91"
   "AF-A0A026W182-F1-model_v4_0001", "0.-.-.-"
   "AF-A0A044RE18-F1-model_v4_0001", "3.4.21.75"
   "AF-A0A072UR65-F1-model_v4_0001", "3.2.1.14"
   "AF-A0A072VIM5-F1-model_v4_0001", "0.-.-.-"
   "AF-A0A075F7E9-F1-model_v4_0001", "2.7.11.1"
   "AF-A0A075F932-F1-model_v4_0001", "0.-.-.-"
   "AF-A0A075FBG7-F1-model_v4_0001", "4.2.3.131;4.2.3.189;4.2.3.190"

Note
----------------
When making predictions, if there are no labels available, you can omit the labels column.

Example Without Labels

.. csv-table::
   :header: "protein_name"
   :widths: 50

   "AF-A0A009IHW8-F1-model_v4_0001"
   "AF-A0A024RXP8-F1-model_v4_0001"
   "AF-A0A024SH76-F1-model_v4_0001"
   "AF-A0A026W182-F1-model_v4_0001"
   "AF-A0A044RE18-F1-model_v4_0001"

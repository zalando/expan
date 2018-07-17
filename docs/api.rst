
===================
API
===================


Architecture
---------------
``core.experiment`` is the most important module to use ExpAn.
It provides interface for running different analysis.

``core.statistics`` provides the underlying statistical functions.
Functionalities in this module includes **bootstrap**, **delta**,
**pooled standard deviation**, **power analysis**, etc.

``core.early_stopping`` provides early stopping algorithms.
It supports **group sequential**, **Bayes factor** and **Bayes precision**.

``core.correction`` implements methods for multiple testing correction.

``core.statistical_test`` holds structures of statistical tests.
You will need the data structure in this module to run an experiment.

``core.results`` holds structures of analysis result.
This will be the running structure of an experiment.

``core.util`` contains supplied common functions used by other modules
such as **generate random data** and **drop nan values**, among many others.

``core.version`` constructs versioning of the package.

``data.csv_fetcher`` reads the raw data and constructs an experiment instance.

``core.binning`` is now DEPRECATED. It implements categorical and numerical **binning algorithms**.
It supports binning implementations which can be applied to unseen data as well.

API
------------

Please visit the :ref:`API <modindex>` list for detailed usage.

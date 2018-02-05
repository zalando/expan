
===================
Functionality
===================

``core.experiment`` is the most important module to use ExpAn.
It provides the analysis functionalities.
Currently we support **delta analysis**, **anomaly detection** and **subgroup analysis**.

``core.statistics`` provides the underlying statistical functions.
These are used by the higher-level experiment module, and can also be used directly.
Functionalities in this module includes **bootstrap**, **delta**, **chi_square**,
**pooled standard deviation** and **power analysis**.

``core.early_stpping`` provides early stopping algorithms.
It supports **group sequential**, **Bayes factor** and **Bayes precision**.

``core.binning`` implements categorical and numerical **binning algorithms**.
It supports binning implementations which can be applied to unseen data as well.

``core.utils`` contains supplied utility functions.
for example, **generate random data** and **drop nan values**, among many other functions.

``core.version`` constructs versioning of the package.

``data.csv_fetcher`` reads the raw data and constructs a ``core.experiment.Experiment`` object.

For a complete list of functionalities, please visit the :ref:`API <modindex>` page.

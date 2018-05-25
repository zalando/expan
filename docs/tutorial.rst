===================
Tutorial
===================

Here is a tutorial to use ExpAn. Let's get started!

Generate demo data
----------------------------
First, let's generate some random data for the tutorial.

.. code-block:: python

	from expan.core.util import generate_random_data
	data, metadata = generate_random_data()

``data`` is a pandas DataFrame.
It must contain a column for entity identifier named **entity**,
a column for variant, and one column per kpi/feature.

``metadata`` is a python dict. It should contain the following keys:

	* ``experiment``: Name of the experiment, as known to stakeholders. It can be anything meaningful to you.
	* ``sources`` (optional): Names of the data sources used in the preparation of this data.
	* ``experiment_id`` (optional): This uniquely identifies the experiment. Could be a concatenation of the experiment name and the experiment start timestamp.
	* ``retrieval_time`` (optional): Time that data was fetched from original sources.
	* ``primary_KPI`` (optional): Primary evaluation criteria.

Currently, ``metadata`` is only used for including more information about the experiment,
and is not taken into consideration for analysis.


Create an experiment
----------------------------------
To use ExpAn for analysis, you first need to create an ``Experiment`` object.

.. code-block:: python

    from expan.core.experiment import Experiment
    exp = Experiment(metadata=metadata)

This ``Experiment`` object has the following parameters:

	* ``metadata``: Specifies an experiment name as the mandatory and data source as the optional fields. Described above.


Create a statistical test
----------------------------------
Now we need a ``StatisticalTest`` object to represent what statistical test to run.
Each statistical test consist of a dataset, one kpi, treatment and control variant names, and the optional features.
Dataset should contain necessary kpis, variants and features columns.

.. code-block:: python

    from expan.core.statistical_test import KPI, Variants, StatisticalTest

    kpi = KPI('normal_same')
    variants = Variants(variant_column_name='variant', control_name='B', treatment_name='A')
    test = StatisticalTest(data=data, kpi=kpi, features=[], variants=variants)


Let's start analyzing!
----------------------------
Running an analysis is very simple:

.. code-block:: python

	exp.analyze_statistical_test(test)

Currently ``analyze_statistical_test`` supports 4 test methods: ``fixed_horizon`` (default), ``group_sequential``, ``bayes_factor`` and ``bayes_precision``.
All methods requires different additional parameters.

If you would like to change any of the default values, just pass them as parameters to delta. For example:

.. code-block:: python

	exp.analyze_statistical_test(test, test_method='fixed_horizon', assume_normal=True, percentiles=[2.5, 97.5])
	exp.analyze_statistical_test(test, test_method='group_sequential', estimated_sample_size=1000)
	exp.analyze_statistical_test(test, test_method='bayes_factor', distribution='normal')

Here is the list of additional parameters.
You may also find the description in our :ref:`API <modindex>` page.

*fixed_horizon* is the default method:

	* ``assume_normal=True``: Specifies whether normal distribution assumptions can be made. A t-test is performed under normal assumption. We use bootstrapping otherwise. Bootstrapping takes considerably longer time than assuming the normality before running experiment. If we do not have an explicit reason to use it, it is almost always better to leave it off.
	* ``alpha=0.05``: Type-I error rate.
	* ``min_observations=20``: Minimum number of observations needed.
	* ``nruns=10000``: Only used if assume normal is false.
	* ``relative=False``: If relative==True, then the values will be returned as distances below and above the mean, respectively, rather than the absolute values.

*group_sequential* is a frequentist approach for early stopping:

	* ``spending_function='obrien_fleming'``: Currently we support only Obrient-Fleming alpha spending function for the frequentist early stopping decision.
	* ``estimated_sample_size=None``: Sample size to be achieved towards the end of experiment. In other words, the actual size of data should be always smaller than estimated_sample_size.
	* ``alpha=0.05``: Type-I error rate.
	* ``cap=8``: Upper bound of the adapted z-score.

*bayes_factor* is a Bayesian approach for delta analysis and early stopping:

	* ``distribution='normal'``: The name of the KPI distribution model, which assumes a Stan model file with the same name exists. Currently we support *normal* and *poisson* models.
	* ``num_iters=25000``: Number of iterations of bayes sampling.
	* ``inference=sampling``: 'sampling' for MCMC sampling method or 'variational' for variational inference method to approximate the posterior distribution.

*bayes_precision* is another Bayesian approach similar as *bayes_factor*:

	* ``distribution='normal'``: The name of the KPI distribution model, which assumes a Stan model file with the same name exists. Currently we support *normal* and *poisson* models.
	* ``num_iters=25000``: Number of iterations of bayes sampling.
	* ``posterior_width=0.08``: The stopping criterion, threshold of the posterior width.
	* ``inference=sampling``: 'sampling' for MCMC sampling method or 'variational' for variational inference method to approximate the posterior distribution.


Interpreting result
-------------------------
The output of the ``analyze_statistical_test`` method is an instance of class :py:class:`core.result.StatisticalTestResult`.
Please refer to the :ref:`API <modindex>` page for result structure as well as descriptions of all fields.
An example of the result is shown below:


.. code-block:: python

    {
        "result": {
            "confidence_interval": [
            {
                "percentile": 2.5,
                "value": 0.1
            },
            {
                "percentile": 97.5,
                "value": 1.1
            }],
            "control_statistics": {
                "mean": 0.0,
                "sample_size": 1000,
                "variance": 1.0
            },
            "delta": 1.0,
            "p": 0.04,
            "statistical_power": 0.8,
            "treatment_statistics": {
                "mean": 1.0,
                "sample_size": 1200,
                "variance": 1.0
            }
        },
        "test": {
            "features": [],
            "kpi": {
                "name": "revenue"
            },
            "variants": {
                "control_name": "control",
                "treatment_name": "treatment",
                "variant_column_name": "variant"
            }
        }
    }


Subgroup analysis
-------------------
Subgroup analysis in ExaAn will select subgroup (which is a segment of data) based on the input argument,
and then perform a regular delta analysis per subgroup as described before.
That is to say, we don't compare between subgroups, but compare treatment with control within each subgroup.

If you wish to perform the test on a specific subgroup,
you can use the ``FeatureFilter`` object:

.. code-block:: python

    feature = FeatureFilter('feature', 'has')
    test = StatisticalTest(data=data, kpi=kpi, features=[feature], variants=variants)


Statistical test suite
----------------------------

It is very common to run a suite of statistical tests.
In this case, you need to create a ``StatisticalTestSuite`` object to represent the test suite.
A ``StatisticalTestSuite`` object consists of a list of ``StatisticalTest`` and a correction method:

.. code-block:: python

	from expan.core.statistical_test import *

	kpi = KPI('normal_same')
	variants = Variants(variant_column_name='variant', control_name='B', treatment_name='A')

	feature_1 = FeatureFilter('feature', 'has')
	feature_2 = FeatureFilter('feature', 'non')
	feature_3 = FeatureFilter('feature', 'feature that only has one data point')

	test_subgroup1 = StatisticalTest(data, kpi, [feature_1], variants)
	test_subgroup2 = StatisticalTest(data, kpi, [feature_2], variants)
	test_subgroup3 = StatisticalTest(data, kpi, [feature_3], variants)

	tests = [test_subgroup1, test_subgroup2, test_subgroup3]
	test_suite = StatisticalTestSuite(tests=tests, correction_method=CorrectionMethod.BH)

And then you can use the ```Experiment``` instance to run the test suite.
Method ``analyze_statistical_test_suite`` has the same arguments as ``analyze_statistical_test``. For example:

.. code-block:: python

	exp.analyze_statistical_test_suite(test_suite)
	exp.analyze_statistical_test_suite(test_suite, test_method='group_sequential', estimated_sample_size=1000)
	exp.analyze_statistical_test_suite(test_suite, test_method='bayes_factor', distribution='normal')


Result of statistical test suite
--------------------------------------

The output of the ``analyze_statistical_test_suite`` method is an instance of class :py:class:`core.result.MultipleTestSuiteResult`.
Please refer to the :ref:`API <modindex>` page for result structure as well as descriptions of all fields.

Following is an example of the analysis result of statistical test suite:

.. code-block:: python

    {
        "correction_method": "BH",
        "results": [
            {
                "test": {
                    "features": [
                        {
                            "column_name": "device_type",
                            "column_value": "desktop"
                        }
                    ],
                    "kpi": {
                        "name": "revenue"
                    },
                    "variants": {
                        "control_name": "control",
                        "treatment_name": "treatment",
                        "variant_column_name": "variant"
                    }
                },
                "result": {
                    "corrected_test_statistics": {
                        "confidence_interval": [
                            {
                                "percentile": 1.0,
                                "value": -0.7
                            },
                            {
                                "percentile": 99.0,
                                "value": 0.7
                            }
                        ],
                        "control_statistics": {
                            "mean": 0.0,
                            "sample_size": 1000,
                            "variance": 1.0
                        },
                        "delta": 1.0,
                        "p": 0.02,
                        "statistical_power": 0.8,
                        "treatment_statistics": {
                            "mean": 1.0,
                            "sample_size": 1200,
                            "variance": 1.0
                        }
                    },
                    "original_test_statistics": {
                        "confidence_interval": [
                            {
                                "percentile": 2.5,
                                "value": 0.1
                            },
                            {
                                "percentile": 97.5,
                                "value": 1.1
                            }
                        ],
                        "control_statistics": {
                            "mean": 0.0,
                            "sample_size": 1000,
                            "variance": 1.0
                        },
                        "delta": 1.0,
                        "p": 0.04,
                        "statistical_power": 0.8,
                        "treatment_statistics": {
                            "mean": 1.0,
                            "sample_size": 1200,
                            "variance": 1.0
                        }
                    }
                }
            },
            {
                "test": {
                    "features": [
                        {
                            "column_name": "device_type",
                            "column_value": "mobile"
                        }
                    ],
                    "kpi": {
                        "name": "revenue"
                    },
                    "variants": {
                        "control_name": "control",
                        "treatment_name": "treatment",
                        "variant_column_name": "variant"
                    }
                },
                "result": {
                    "corrected_test_statistics": {
                        "confidence_interval": [
                            {
                                "percentile": 1.0,
                                "value": -0.7
                            },
                            {
                                "percentile": 99.0,
                                "value": 0.7
                            }
                        ],
                        "control_statistics": {
                            "mean": 0.0,
                            "sample_size": 1000,
                            "variance": 1.0
                        },
                        "delta": 1.0,
                        "p": 0.02,
                        "statistical_power": 0.8,
                        "stop": false,
                        "treatment_statistics": {
                            "mean": 1.0,
                            "sample_size": 1200,
                            "variance": 1.0
                        }
                    },
                    "original_test_statistics": {
                        "confidence_interval": [
                            {
                                "percentile": 2.5,
                                "value": 0.1
                            },
                            {
                                "percentile": 97.5,
                                "value": 1.1
                            }
                        ],
                        "control_statistics": {
                            "mean": 0.0,
                            "sample_size": 1000,
                            "variance": 1.0
                        },
                        "delta": 1.0,
                        "p": 0.04,
                        "statistical_power": 0.8,
                        "stop": true,
                        "treatment_statistics": {
                            "mean": 1.0,
                            "sample_size": 1200,
                            "variance": 1.0
                        }
                    }
                }
            }
        ]
    }


That's it!

For API list and theoretical concepts, please read the next sections.


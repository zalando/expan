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

``metadata`` is a python dict. Keys of it can contain:

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
    exp = Experiment(data=data, metadata=metadata)

This ``Experiment`` object has the following parameters:

	* ``data``: A data you want to run experiment for. An example of the data structure. Described above.
	* ``metadata``: Specifies an experiment name as the mandatory and data source as the optional fields. Described above.


Create a statistical test
----------------------------------
Now we need a ``StatisticalTest`` object to represent what statistical test to run.
Each statistical test consist of one kpi, treatment and control variant names, and the optional features.

.. code-block:: python

    from expan.core.statistical_test import KPI, Variants, StatisticalTest

    kpi = KPI('normal_same')
    variants = Variants(variant_column_name='variant', control_name='B', treatment_name='A')
    test = StatisticalTest(kpi=kpi, features=[], variants=variants)

If you wish to perform the test on a specific subgroup,
you can use the ``FeatureFilter`` object:

.. code-block:: python

    feature = FeatureFilter('feature', 'has')
    test = StatisticalTest(kpi=kpi, features=[feature], variants=variants)

Subgroup analysis will be covered in later sections.


Let's start analyzing!
----------------------------
Running an analysis is very simple:

.. code-block:: python

	exp.analyze_statistical_test(test)

Currently ``delta`` supports 4 methods: ``fixed_horizon`` (default), ``group_sequential``, ``bayes_factor`` and ``bayes_precision``.
All methods requires different additional parameters.

If you would like to change any of the default values, just pass them as parameters to delta. For example:

.. code-block:: python

	exp.analyze_statistical_test(test, method='fixed_horizon', assume_normal=True, percentiles=[2.5, 97.5])
	exp.analyze_statistical_test(test, method='group_sequential', estimated_sample_size=1000)
	exp.analyze_statistical_test(test, method='bayes_factor', distribution='normal')

Here is the list of addtional parameters.
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


Statistical test suite
----------------------------





Subgroup analysis
-------------------
Subgroup analysis in ExaAn will select subgroup (which is a segment of data) based on the input argument,
and then perform a regular delta analysis per subgroup as described before.
That is to say, we don't compare between subgroups, but compare treatment with control within each subgroup.


An example is provided below.

.. code-block:: python

	dimension_to_bins = {"treatment_start_time": [
	    Bin("numerical", 0, 5, True, False),
	    Bin("numerical", 5, 10, True, False)]
	}
	exp.sga(dimension_to_bins)

And the result of subgroup analysis is:

.. code-block:: python




That's it! Try it out for yourself: `<github.com/zalando/expan>`_


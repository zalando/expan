===================
Tutorial
===================

Here we will provide a tutorial of using ExpAn.

First, let's generate some random data.

.. code-block:: python

	from expan.core.util import generate_random_data
	data, metadata = generate_random_data()

``data`` is a pandas DataFrame.
It must contain a column **entity**, a column **variant**, then one column per kpis you defined.
You can check the example structure of ``data`` aby looking at:

.. code-block:: python

	data.head()

``metadata`` is a python dict.
Key of this dict should contain:

	* ``experiment``: Name of the experiment, as known to stakeholders. It can be anything meaningful to you.
	* ``sources`` (optional): Names of the data sources used in the preparation of this data.
	* ``experiment_id`` (optional): This uniquely identifies the experiment. Could be a concatenation of the experiment name and the experiment start timestamp.
	* ``retrieval_time`` (optional): Time that data was fetched from original sources.
	* ``primary_KPI`` (optional): Primary evaluation criteria.


Constructing an experiment
----------------------------
To use ExpAn for analysis, you first need to construct an ``Experiment`` object.

.. code-block:: python

	from expan.core.experiment import Experiment
	exp = Experiment(control_variant_name='A',
	                 data=data,
	                 metadata=metadata,
	                 report_kpi_names=['derived_kpi_one'],
	                 derived_kpis=[{'name':'derived_kpi_one','formula':'normal_same/normal_shifted'}])

This ``Experiment`` object has the following parameters:

	* ``control_variant_name``: Indicates which of the variants is to be considered as a baseline (a.k.a. control).
	* ``data``: A data you want to run experiment for. An example of the data structure. Described above.
	* ``metadata``: Specifies an experiment name as the mandatory and data source as the optional fields. Described above.
	* ``report_kpi_names``: A list of strings specifying desired kpis to analyse (empty list by default).
	* ``derived_kpis``: Each derived kpi is defined as a dictionary structured by *{'name': <name_of_the_kpi>, 'formula': <formula_to_compute_kpi>}*. Then **derived_kpis** is a list of such dictionaries if more than one derived_kpi is wanted (empty dict by default). *<name_of_the_kpi>* is name of the kpi. *<formula_to_compute_kpi>* is the formula to calculate the desired kpi. You can find the example described above.

**NOTE 1**. You should be careful specifying the correct structure to the derived_kpis dictionary including keys *'name'* and *'formula'*. Otherwise, construction of ``Experiment`` object will raise an exception.

**NOTE 2**. Specify the derived kpi name in the *report_kpi_names* if you want to see the results for it too.

**NOTE 3**. Wrong input structures (e.g. missing derived_kpis dictionary keys or incorrect kpi keys) will raise an exception.


In our generated demo data we have two variants and one of them is the baseline (aka. control):

.. code-block:: python

	print('Variants: {}'.format(exp.variant_names))
	print('Control or baseline variant: {}'.format(exp.control_variant_name))

.. code-block:: console

   >>> Variants: {'A', 'B'}
   >>> Control or baseline variant: A


Now we can start analysing!
----------------------------
Analysis is as simple as:

.. code-block:: python

	exp.delta()

Currently ``delta`` supports 4 methods: *fixed_horizon* (default), *group_sequential*, *bayes_factor* and *bayes_precision*. All methods requires different additional parameters.

If you would like to change any of the default values, just pass them as parameters to delta. For example:

.. code-block:: python

	exp.delta(method='fixed_horizon', assume_normal=True, percentiles=[2.5, 97.5])
	exp.delta(method='group_sequential', estimated_sample_size=1000)
	exp.delta(method='bayes_factor', distribution='normal')

Here is the list of each of the addtional parameters.
You may also find the description in our :ref:`API <modindex>` page.

*fixed_horizon* is the default method:

	* ``assume_normal=True``: Specifies whether normal distribution assumptions can be made. A t-test is performed under normal assumption. We use bootstrapping otherwise. Bootstrapping takes considerably longer time than assuming the normality before running experiment. If we do not have an explicit reason to use it, it is almost always better to leave it off.
	* ``percentiles=[2.5, 97.5]``: A list of percentile values for confidence bounds.
	* ``min_observations=20``: Minimum number of observations needed.
	* ``nruns=10000``: Only used if assume normal is false.
	* ``relative=False``: If relative==True, then the values will be returned as distances below and above the mean, respectively, rather than the absolute values.
	* ``multi_test_correction=False``: Initiate multiple correction (Bonferroni correction is supported).

*group_sequential* is a frequentist approach for early stopping:

	* ``spending_function='obrien_fleming'``: Currently we support only Obrient-Fleming alpha spending function for the frequentist early stopping decision.
	* ``estimated_sample_size=None``: Sample size to be achieved towards the end of experiment. In other words, the actual size of data should be always smaller than estimated_sample_size.
	* ``alpha=0.05``: Type-I error rate.
	* ``cap=8``: Upper bound of the adapted z-score.
	* ``multi_test_correction=False``: Initiate multiple correction (Bonferroni correction is supported).

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
The output of the ``delta`` method has the following structure:

.. code-block:: python

	{
	  "warnings": [
	    "kpi: derived_kpi_one, variant: B: Sample variances differ too much to assume that population variances are equal."
	  ],
	  "errors": [],
	  "expan_version": "0.6.2",
	  "control_variant": "A",
	  "kpis": [
	    {
	      "name": "derived_kpi_one",
	      "variants": [
	        {
	          "name": "A",
	          "delta_statistics": {
	            "stop": true,   # only available for early-stopping deltas: group_sequential, bayes_factor and bayes_precision
	            "delta": 0.0,
	            "confidence_interval": [
	              {
	                "percentile": 2.5,
	                "value": -8.396765530428699
	              },
	              {
	                "percentile": 97.5,
	                "value": 3.5794735964894677
	              }
	            ],
	            "treatment_sample_size": 6108,
	            "control_sample_size": 6108,
                "control_variance": 33019.25141863071,  # only available for frequentist deltas: fixed_horizon, group_sequential
                "treatment_variance": 33019.25324523451, # only available for frequentist deltas: fixed_horizon, group_sequential
	            "treatment_mean": -4.572524000045541,
	            "control_mean": -4.572524000045541,
	            "number_of_iterations": 25000, # only available for Bayesian deltas: bayes_factor and bayes_precision
	            "statistical_power": 0.050000000000000044
	          }
	        },
	        {
	          "name": "B",
	          "delta_statistics": {
	            "stop": true,
	            "delta": 4.564575415240889,
	            "confidence_interval": [
	              {
	                "percentile": 2.5,
	                "value": -2.8506067127900847
	              },
	              {
	                "percentile": 97.5,
	                "value": 8.497896742163277
	              }
	            ],
	            "treatment_sample_size": 3892,
	            "control_sample_size": 6108,
	            "treatment_mean": -0.007948584804651233,
	            "control_mean": -4.572524000045541,
	            "number_of_iterations": 25000,
                "control_variance": 33019.25141863071,
                "treatment_variance": 33019.25324523451,
	            "statistical_power": 0.46900387352149797
	          }
	        }
	      ]
	    }
	  ]
	}

(note that some values shown here can be made up)


The corresponding fields are:

	* ``treatment_mean``: the mean of the treatment group.
	* ``control_mean``: the mean of the control group.
	* ``treatment_variance``: the variance of the treatment group.
	* ``control_variance``: the variance of the control group.
	* ``control_sample_size``: the sample size for the control group.
	* ``treatment_sample_size``: the sample size for the treatment group.
	* ``delta``: the difference between the ``treatment_mean`` and ``control_mean``.
	* ``confidence_interval``: the confidence interval: ``percentile`` - lower percentile and upper percentile; ``value`` - value for each percentile.
	* ``number_of_iterations``: number of iterations used for bayes sampling for *bayes_factor* and *bayes_precision* methods.
	* ``stop``: flag indicating whether the experiment can be stopped. This flag exists for early stopping methods.
	* ``statistical_power``: the value of statistical power --- that is, the probability of a test to detect an effect, if the effect actually exists.



Binning
-------------------
You can use the Binning module to group data into subsets, i.e., assign each data into a corresponding ``Bin`` object. We will explain respectively in the next few sections.

Create bin object directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you already know the set of bins you want to put data into. You can initialize a ``bin`` object directly.

The first argument is the id of the bin. This might not be useful for your application but serves as a technical identifier.
The second argument is the type of the bin. This can either be "numerical" or "categorical". Depending on the type, you should pass coresponding representation object as the third argument.

.. code-block:: python

	from expan.core.binning import *

Create a numerical bin from value 0 (inclusive) to 10 (exclusive).

.. code-block:: python

	Bin("numerical", 0, 10, True, False)

will output:

.. code-block:: console

	bin: [0, 10)

Create a categorical bin which contains categories of "a" and "b".

.. code-block:: python

	Bin("categorical", ["a", "b"])

will output:

.. code-block:: console

	bin: ['a', 'b']

Create bin object automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given a number of bins, you can also create a list of bins from data by using the method ``create_bins(data, n_bins)``.

It will create n_bins ``Bin`` ojbects, which separates ``data`` as equally as possible. This method will also automatically detects numerical or categorical data, and creates corresponding bin representations.


.. code-block:: python

	data_control = exp.data[exp.data.variant == 'A']
	data_treatment = exp.data[exp.data.variant == 'B']

.. code-block:: python

	n_bins = 10
	create_bins(data_control.normal_same, n_bins)

will output:

.. code-block:: python

	[
	  bin: [-3.83665554846, -1.25906491145),
	  bin: [-1.25906491145, -0.804751813719),
	  bin: [-0.804751813719, -0.489466995342),
	  bin: [-0.489466995342, -0.226662203724),
	  bin: [-0.226662203724, 0.0239463824493),
	  bin: [0.0239463824493, 0.276994331119),
	  bin: [0.276994331119, 0.551060124216),
	  bin: [0.551060124216, 0.868798338306),
	  bin: [0.868798338306, 1.30062540106),
	  bin: [1.30062540106, 4.47908425103]
	]

Assign data to bins
~~~~~~~~~~~~~~~~~~~~~
We can use the method ``apply(data)`` of the ``Bin`` object to assign data to one of the given bins.
This method will return a subset of input data which belongs to this bin.
It will return ``None`` if there is no data matched.

.. code-block:: python

	bin = Bin("numerical", 0, 10, True, False)
	bin.apply(data_control.normal_same)

Applying bin to data in variant A will result in:

.. code-block:: python

	4       1.112634
	6       0.085595
	10      0.335054
	13      0.542203
	15      0.002232
	19      0.467690
	21      1.171102
	23      1.289203
	27      0.141980
	28      0.313723
	31      1.345935
	37      2.418778
	41      0.288028
	44      0.411566
	46      1.120967
	47      0.805575
	48      0.975823
	49      0.008858
	54      1.352039
	57      2.159121
	58      0.091315
	61      1.637082
	63      0.735269
	66      1.030250
	71      0.644690
	77      0.723038
	78      0.085513
	83      1.889279
	84      0.238171
	89      0.580568
	          ...
	9873    0.030269
	9875    0.863606
	9876    0.524865
	9880    0.008274
	9891    0.395712
	9900    1.168769
	9901    0.055230
	9903    0.192369
	9908    0.010693
	9909    0.354407
	9910    0.853060
	9914    0.492523
	9918    0.502002
	9924    1.096724
	9925    0.688108
	9934    0.367047
	9935    0.279812
	9936    0.445043
	9945    0.876760
	9948    0.261577
	9954    1.601119
	9955    1.797017
	9959    0.542985
	9969    0.206816
	9973    1.589447
	9980    0.130357
	9982    0.377618
	9985    0.193655
	9986    0.055740
	9987    0.664763
	Name: normal_same, dtype: float64


Similarly, applying bin to data in variant B will result in different result:

.. code-block:: python

  bin.apply(data_treatment.normal_same)

.. code-block:: python

	2       0.388819
	8       0.772848
	9       0.783160
	11      0.564789
	20      1.310606
	25      0.600733
	30      0.608267
	33      1.360168
	35      0.849585
	38      1.495458
	43      0.444854
	51      0.977872
	55      2.099408
	69      1.132805
	70      1.597397
	74      0.079915
	75      0.320930
	86      0.220631
	88      0.324758
	92      1.638961
	104     1.277857
	107     1.498012
	115     1.344854
	118     2.120994
	127     0.059905
	139     2.254038
	156     0.079048
	161     0.150602
	165     0.090310
	170     0.947512
	          ...
	9862    0.725924
	9863    1.492610
	9864    0.908889
	9883    1.138699
	9885    0.167043
	9886    0.285282
	9887    0.322020
	9894    2.127297
	9897    1.896604
	9911    1.127925
	9913    0.499415
	9915    0.327819
	9927    0.729370
	9928    0.887623
	9937    0.278923
	9938    0.729843
	9940    0.201785
	9943    1.338250
	9957    0.544323
	9958    0.858663
	9971    0.290580
	9972    1.081581
	9977    0.460328
	9981    0.084888
	9983    0.443676
	9984    0.338594
	9989    1.544333
	9993    0.672613
	9996    0.792395
	9997    0.994518
	Name: normal_same, dtype: float64



Subgroup analysis
-------------------
Subgroup analysis in ExaAn will select subgroup (which is a segment of data) based on the input argument, and then perform a regular delta analysis per subgroup as described before.
That is to say, we don't compare between subgroups, but compare treatment with control within each subgroup.

The input argument is a python dict, which maps feature name (key) to a list of ``Bin`` objects (value).
This dict defines how and on which feature to perform the subgroup split.
The returned value of subgroup analysis will be the result of regular delta analysis per subgroup.

An example is provided below.

.. code-block:: python

	dimension_to_bins = {"treatment_start_time": [
	    Bin("numerical", 0, 5, True, False),
	    Bin("numerical", 5, 10, True, False)]
	}
	exp.sga(dimension_to_bins)

And the result of subgroup analysis is:

.. code-block:: python

	[
	  {
	    'dimension': 'treatment_start_time',
	    'segment': '[0, 5)'
	    'result': {
	      'warnings': ['kpi: derived_kpi_one, variant: B: Sample variances differ too much to assume that population variances are equal.']
	      'control_variant': 'A',
	      'errors': [],
	      'expan_version': '0.6.2',
	      'kpis': [
	        {
	          'name': 'derived_kpi_one',
	          'variants': [
	            {
	              'name': 'B',
	              'delta_statistics': {
	                'control_mean': -0.32639393302612346,
	                'control_sample_size': 3076,
	                'delta': 0.3204731468864935,
	                'statistical_power': 0.095063282824786377,
	                'treatment_mean': -0.005920786139629961,
	                'treatment_sample_size': 1930,
	              	'confidence_interval': [
	              	  {'percentile': 2.5, 'value': -1.5569210692070499},
	                  {'percentile': 97.5, 'value': 2.1978673629800363}
	              	]
	              }
	            },
	            {
	              'name': 'A',
	              'delta_statistics': {
	                'control_mean': -0.32639393302612346,
	                'control_sample_size': 3076,
	                'delta': 0.0,
	                'statistical_power': 0.050000000000000044,
	                'treatment_mean': -0.32639393302612346,
	                'treatment_sample_size': 3076,
	                'confidence_interval': [
	                  {'percentile': 2.5,'value': -2.1025221680926345},
	                  {'percentile': 97.5, 'value': 2.102522168092634}
	                ]
	              }
	            }
	          ]
	        }
	      ]
	    }
	  },
	  {
	    'dimension': 'treatment_start_time',
	    'segment': '[5, 10)'
	    'result': {
	      'warnings': ['kpi: derived_kpi_one, variant: B: Sample variances differ too much to assume that population variances are equal.']
	      'control_variant': 'A',
	      'errors': [],
	      'expan_version': '0.6.2',
	      'kpis': [
	        {
	          'name': 'derived_kpi_one',
	          'variants': [
	            {
	              'name': 'B',
	              'delta_statistics': {
	                'control_mean': 3.379775978641749,
	                'control_sample_size': 3032,
	                'delta': -3.389734477426074,
	                'statistical_power': 0.59356839479094403,
	                'treatment_mean': -0.009958498784324924,
	                'treatment_sample_size': 1962},
	                'confidence_interval': [
	                  {'percentile': 2.5,'value': -6.9215226752839172},
	                  {'percentile': 97.5, 'value': 0.14205372043176823}
	                ]
	            },
	            {
	              'name': 'A',
	              'delta_statistics': {
	                'control_mean': 3.379775978641749,
	                'control_sample_size': 3032,
	                'delta': 0.0,
	                'statistical_power': 0.050000000000000044,
	                'treatment_mean': 3.379775978641749,
	                'treatment_sample_size': 3032},
	                'confidence_interval': [
	                  {'percentile': 2.5, 'value': -4.017338312084763},
	                  {'percentile': 97.5, 'value': 4.0173383120847621}
	                ]
	            }
	          ]
	        }
	      ],
	    },
	  }
	]

As you can see, the hierarchy of the result of subgroup analysis is the following:

.. code-block:: console

	-subgroups
	  -kpis
	    -variants


That's it! Try it out for yourself: `<github.com/zalando/expan>`_


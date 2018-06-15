==========
Glossary
==========


Assumptions used in analysis
------------------------------------

1. Sample-size estimation

  * Treatment does not affect variance
  * Variance in treatment and control is identical
  * Mean of delta is normally distributed

2. Welch t-test

  * Mean of means is t-distributed (or normally distributed)

3. In general

  * Sample represents underlying population
  * Entities are independent


Derived KPIs, such as conversion rate
-------------------------------------
For each user, we have their number of orders and their number of sessions.
We estimate the orders-per-session ("conversion rate") by computing
the total number of orders across all users and divide that by
the total number of sessions across all users.
Equivalently, we can use the ratio of the means:

.. math::

    \overline{CR} = \mbox{estimated conversion rate} = \frac{ \sum_{i=1}^n o_i }{ \sum_{i=1}^n s_i } = \frac{ \frac1{n} \sum_{i=1}^n o_i }{ \frac1{n} \sum_{i=1}^n s_i } = \frac{\bar{o}}{\bar{s}}

To calculate the variance of this estimate, and therefore apply a t-test, we need to compute the variance of this
estimator. If we used the same data again, but randomly reassigned every user to a group (treatment or control),
and recomputed :math:`\overline{CR}` many times, how would this estimate vary?

For each user, the "error" (think linear regression) is:

.. math::

    e_i = o_i - s_i{\cdot}\overline{CR}

The subtracted portion :math:`(-s_i \cdot \overline{CR})` is essentially non-random for our purposes,
allowing us to say - to a very good approximation - that :math:`Var[o_i]=Var[e_i]`.
Also, the **e** vector will have mean zero by construction.

Therefore, as input to the pooled variance calculation, we use this as the variance estimate:

.. math::

    \hat{Var}\left[ \frac{ o_i }{ \bar{s} } \right]
    = \hat{Var}\left[ \frac{ e_i }{ \bar{s} } \right]
    = \frac1{n-1} \sum_{i=1}^n \left(\frac{e_i - \bar{e}}{\bar{s}}\right)^2
    = \frac1{n-1} \sum_{i=1}^n \left(\frac{e_i}{\bar{s}}\right)^2

The variances are calculated as above for both the control and the treatment and fed into
a pooled variance calculation as usual for a t-test.

See the test named ``test_using_lots_of_AA_tests()`` within ``expan/tests/test_derived.py``
for a demonstration of how this method gives a uniform p-value under the null;
this confirms that the correct error rate is maintained.

Early stopping
------------------------------------

Given samples x from treatment group, samples y from control group, we want to know whether there is a significant difference between the means :math:`\delta=\mu(y)−\mu(x)`.
To save the cost of long-running experiments, we want to stop the test early if we are already certain that there is a statistically significant result.

You can find links to our detailed documentations for
`concept of early stopping <https://github.com/shansfolder/AB-Test-Early-Stopping/blob/master/docs/EarlyStoppingConcept/EarlyStoppingConcept.pdf>`_ and
`early stopping methods we investigated <https://github.com/shansfolder/AB-Test-Early-Stopping/blob/master/docs/EvaluateEarlyStopping/EvaluatingEarlyStopping.pdf>`_.


Subgroup analysis
------------------------------------
Subgroup analysis in ExpAn will select subgroup (which is a segment of data) based on the input argument, and then perform a regular delta analysis per subgroup as described before.

That is to say, we don't compare between subgroups, but compare treatment with control within each subgroup.

To support automatic detection of those interesting subgroups, also known as Heterogeneous Treatment Effect, is under planning.

Multiple testing problem
------------------------------------
Multiple testing problem occurs when one considers a set of statistical inferences simultaneously. Consider a set of :math:`20` hypothesis that you wish to test at the
significance level of :math:`0.05`.
What is the probability of observing at least one significant result just due to chance?

:math:`\Pr \textrm{(at least one significant result)} = 1 - \Pr \textrm{(no significant results)} = 1 - (1 - 0.05)^{20} \approx 0.64`

With :math:`20` tests being considered, we have a :math:`64\%` chance of observing at least one significant result, even if all of the tests are actually not significant.
Methods for dealing with multiple testing frequently call for adjusting :math:`\alpha` in some way, so that the probability of observing at least one significant result due to chance
remains below your desired significance level.

ExpAn allows you to correct :math:`\alpha` by setting ``multi_test_correction`` flag to True. It uses the simplest, but quite conservative Bonferroni correction method.
The Bonferroni correction sets the significance cut-off at :math:`\frac{\alpha}{n}` where :math:`n` is the number of tests.
With multiple correction of :math:`25` experiments your adjusted percentiles change from :math:`[2.5, 97.5]` to :math:`[0.1, 99.9]`.

We understand that the Bonferroni correction may be very conservative and the correction comes at the cost of increasing the probability of producing type II errors (false negatives),
that's why we plan to make updates for supporting more clever correction methods like Benjamini-Hochberg or Benjamini-Krieger-Yekutieli, which will come soon.

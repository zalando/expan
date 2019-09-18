==========
Glossary
==========


Assumptions used in analysis
------------------------------------

1. Sample-size estimation

  * Treatment does not affect variance
  * Variance in treatment and control is identical
  * Mean of delta is normally distributed

2. Equal or unequal sample sizes, equal variance t-test

  * Mean of means is t-distributed (or normally distributed)
  * Variance of two distributions are same (so the variance of two groups of sample should be similar)

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

As a side comment, you might be tempted to compute the ratio for each individual, :math:`\frac{o_i}{s_i}`,
and compute the mean of those ratios, :math:`\overline{\left(\frac{o}{s}\right)_i}`.
The problem with this is that it's an estimator with low accuracy; more formally, its variance is large.
Intuitively, we want to compute a mean by giving greater weight to ratios which have more sessions;
this is how we derive the formula for :math:`\overline{CR}` above.

To calculate the variance of this estimate, and therefore apply a t-test, we need to compute the variance of this
estimator. If we used the same data again, but randomly reassigned every user to a group (treatment or control),
and recomputed :math:`\overline{CR}` many times, how would this estimate vary?

We model that the :math:`s_i` are given (i.e. non-random), and the :math:`o_i` are random variables
whose distribution is a function of :math:`s_i`.

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

Finally, this method doesn't suffer from the problem described in
`this blog post <https://towardsdatascience.com/the-second-ghost-of-experimentation-the-fallacy-of-session-based-metrics-fb65006d30ff>`_.
In our notation, :math:`o_i` is the sum of the orders for all session for user :math:`i`.
The method criticized in that blog post is to compute the variance estimate across every session, i.e. ignoring :math:`o_i` and instead using
the per-session orders individually.
That is problematic because it ignores the fact that the sessions for a given user may be correlated with each other.
Our approach is different and follows the linear regression procedure closely,
and therefore is more robust to these issues.

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

ExpAn allows you to control the correction method for your set of statistical tests (statistical test suite) yourself.
There are three options for the correction method:

* **CorrectionMethod.BONFERRONI**- strict `Bonferroni correction <https://en.wikipedia.org/wiki/Bonferroni_correction>`_ which controls the `family-wise error rate <https://en.wikipedia.org/wiki/Family-wise_error_rate>`_.

* **CorrectionMethod.BH** - correction by Benjamini-Hochberg: less strict and more powerful correction method which decreases the `false discovery rate <https://en.wikipedia.org/wiki/False_discovery_rate>`_.

* **CorrectionMethod.NONE** - no correction is used. Even this option is available in ExpAn we strongly recommend to do not neglect the importance of correction for multiple testing and always correct for multiple testing using Benjamini-Hochberg correction, as a default one (as currently set up in ExpAn).

Correction is performed per each statistical test suite, but you can use the correction methods separately
by calling ``benjamini_hochberg(false_discovery_rate, original_p_values)`` or
``bonferroni(false_positive_rate, original_p_values)`` providing corresponding p-values for the correction.

Read more about each correction method:

* `Benjamini-Hochberg <https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure>`_ or original paper "Hochberg, Y., and A. C. Tamhane. Multiple Comparison Procedures."

* `Bonferroni <https://en.wikipedia.org/wiki/Bonferroni_correction>`_


Chi-square test (Multinomial Goodness of Fit Test).
------------------------------------
In ExpAn we have the possibility to conduct multinomial goodness of fit test (chi-square test).
The test is applied when you have one categorical variable from a single population.
It is used to determine whether sample data are consistent with a hypothesized distribution
(allocation of traffic or split percentage).

This test, in our case, is used to check the variant split based on the claimed percentage.
For example, we want 50% of the users to be exposed to control variant (for example, green checkout button)
and 50% of the users to be exposed to treatment variant (for example, yellow checkout button).
We conduct a random assignment of variants and would like to check whether the random assignment did the right job and
we've got the correct split of the variants. We would also like to know whether the variant split consistent with the specified
percentage after the outlier filtering as well.

The Ho is: the data are consistent with a specified distribution (or the variant split corresponds to the expected percentage)
The Ha is: the data are not consistent with a specified distribution (or the variant split do not correspond to the expected percentage)
Typically, the null hypothesis (Ho) specifies the proportion of observations at each level of the categorical variable.
The alternative hypothesis (Ha) is that at least one of the specified proportions is not true.

Multinomial goodness of fit test is described with one intuitive formula:

.. math::

    {\chi}^2_{K-1} = \sum_{i=1}^{K} \frac{(O_i - E_i)^2}{E_i}


Here :math:`O` denotes the observed number of users buckets in bucket :math:`i`, and :math:`E` denotes the expected
number of users bucketed in each bucket. :math:`K` - overall number of buckets. The statistics capture how much each bucket deviates from the expected value,
and the summation captures the overall deviation.
`Source <https://blog.twitter.com/engineering/en_us/a/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests.html>`_

We use 0.05 significance level as the default one.
We compute p-value - the probability of observing a sample statistics as extreme as the test statistic - and compare it to the significance level.
We reject the null hypothesis when the p-value is less than the significance level.

We can use this test to check the correct split for the subgroups as well.

* Multiple testing problem for chi-square testing

Since chi-square testing is also a hypothesis testing, you need to keep in mind that multiple chi-square testing brings
the problem of increasing of false positives rate described in the previous section.
Let say, you want to test the correctness of your variants split 5 times at different times with 0.05 alpha.
For 5 tests your alpha is no longer 0.05, but :math:`1 - (1 - 0.05)^{5} \approx 0.23`. Correction for multiple chi-square testing is needed here.
In this case, you can run several chi-square tests and collect p-values, сorrect p-values with one of our correction
methods (CorrectionMethod.BONFERRONI, CorrectionMethod.BH) to get new corrected alpha, and make a decision
about correctness of the variants splits using that new alpha.

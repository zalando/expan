=====
Usage
=====

To use ExpAn in a project::

    import expan

Some mock-up data
-----------------

::

    from expan.core.experiment import Experiment
    from expan.core.util import generate_random_data

    exp = Experiment('B', *generate_random_data())
    exp.delta()


Per-entity ratio vs. ratio of totals
------------------------------------

There are two different definitions of a ratio metric (think of e.g. conversion rate, which is the ratio between the number of orders and the number of visits): 1) one that is based on the entity level or 2) ratio between the total sums, and **ExpAn** supports both of them.

In a nutshell, one can reweight the individual **per-entity ratio** to calculate the **ratio of totals**. This enables to use the existing ``statistics.delta()`` function to calculate both ratio statistics (either using normal assumtion or bootstraping).

Calculating the conversion rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example let's look at how to calculate the conversion rate, which might be typically defined per-entity as the average ratio between the number of orders and the number of visits:

.. math::

	\overline{CR}^{(pe)} = \frac{1}{n} \sum_{i=1}^n CR_i = \frac{1}{n} \sum_{i=1}^n \frac{O_i}{V_i}

The ratio of totals is a reweighted version of :math:`CR_i` to reflect not the entities' contributions (e.g. contribution per custormer) but overall equal contributions to the conversion rate, which can be formulated as:

.. math::

	CR^{(rt)} = \frac{\sum_{i=1}^n O_i}{\sum_{i=1}^n V_i}

Overall as reweighted Individual
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can calculate the :math:`CR^{(rt)}` from the :math:`\overline{CR}^{(pe)}` using the following weighting factor (easily proved by paper and pencile):

.. math::

	CR^{(rt)} = \frac{1}{n} \sum_{i=1}^n \alpha_i \frac{O_i}{V_i}

with

.. math::

	\alpha_i = n \frac{V_i}{\sum_{i=1}^n V_i}

Weighted delta function
^^^^^^^^^^^^^^^^^^^^^^^

To have such functionality as a more generic approach in **ExpAn**, we can introduce a *weighted delta* function. Its input are

- The per-entity metric, e.g. :math:`O_i/V_i`
- A reference metric, on which the weighting factor is based, e.g. :math:`V_i`

With this input it calculates :math:`\alpha` as described above and outputs the result of ``statistics.delta()``.

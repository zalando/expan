=====
Usage
=====

To use ExpAn in a project::

    import expan

Some mock-up data
-----------------

::

    from expan.core.experiment import Experiment
    from tests.tests_core.test_data import generate_random_data

    exp = Experiment('B', *generate_random_data())
    exp.delta()


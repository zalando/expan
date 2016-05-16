ExpAn: Experiment Analysis
==========================

.. image:: https://img.shields.io/travis/zalando/expan.svg
        :target: https://travis-ci.org/zalando/expan
        :alt: Build status

.. image:: https://img.shields.io/pypi/v/expan.svg
        :target: https://pypi.python.org/pypi/expan
        :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/status/expan.svg
   :target: https://pypi.python.org/pypi/expan
   :alt: Development Status

.. image:: https://img.shields.io/pypi/pyversions/expan.svg
   :target: https://pypi.python.org/pypi/expan
   :alt: Python Versions


.. image:: https://img.shields.io/pypi/dw/expan.svg
        :target: https://pypi.python.org/pypi/expan/
        :alt: PyPI Downloads

.. image:: https://img.shields.io/pypi/l/expan.svg
        :target: https://pypi.python.org/pypi/expan/
        :alt: License

A/B tests (a.k.a. Randomized Controlled Trials or Experiments) have been widely
applied in different industries to optimize business processes and user
experience. ExpAn (**Exp**\ eriment **An**\ alysis) is a Python library
developed for the statistical analysis of such experiments and to standardise
the data structures used.

The data structures and functionality of ExpAn are generic such that they can be
used by both data scientists optimizing a user interface and biologists
running wet-lab experiments. The library is also standalone and can be
imported and used from within other projects and from the command line.

Major statistical functionalities include:

- **feature check**
- **delta**
- **subgroup analysis**
- **trend**

Table of Contents
=================

-  `Quick start <#quick-start>`__

   -  `Install <#install>`__
   -  `Some mock-up data <#some-mock-up-data>`__

-  `Further documentation <#further-documentation>`__
-  `How to contribute <#how-to-contribute>`__

   -  `Style guide <#style-guide>`__
   -  `Branching / Release <#branching--release>`__
   -  `Versioning <#versioning>`__
   -  `Bumping Version <#bumping-version>`__
   -  `Travis CI and PyPI deployment <#travis-ci-and-pypi-deployment>`__
   -  `TODO <#todo>`__

-  `License <#license>`__

Quick start
===========

Install
-------

To install you can simply run (pip >= 8.1.1 and setuptools >= 21.0.0 are required):

::

	pip install expan


An alternative way to install is it to clone the repo and run:

::

    python2 setup.py build
    python2 setup.py install

And to test run:

::

    python2 setup.py test

Some mock-up data
-----------------

::

    from expan.experiment import Experiment
    from tests.test_data import generate_random_data

    exp = Experiment('B', *generate_random_data())
    exp.delta()


Further documentation
=====================

`ExpAn Description <https://github.com/zalando/expan/blob/master/ExpAn-Description.mediawiki>`__ - details about the concept of the library and data structures.

`ExpAn Introduction <https://github.com/zalando/expan/blob/dev/ExpAn-Intro.ipynb>`__ - a full jupyter (iPython) notebook. You can view it as slides with `jupyter <http://jupyter.org>`__:

::

    sh serve_intro_slides

Alternatives
============

There may be alternative libraries providing similar functionality, and these
should be collected here. Very incomplete list so far...

abba (https://github.com/thumbtack/abba)
----

Not an alternative, the Python part of this is simply a collection of some functions to handle binomial distributions.


How to contribute
=================

Style guide
-----------

We follow `PEP8 standards <https://www.python.org/dev/peps/pep-0008>`__
with the following exceptions:

- Use *tabs instead of spaces* - this allows all individuals to have visual depth of indentation they prefer, without changing the source code at all, and it is simply smaller

Branching / Release
-------------------

We currently use the gitflow workflow. Feature branches are created from
and merged back to the ``dev`` branch, and the ``master`` branch stores
snapshots/releases of the ``dev`` branch.

See also the much simpler github flow
`here <http://scottchacon.com/2011/08/31/github-flow.html>`__

Versioning
----------

**For the sake of reproducibility, always be sure to work with a release
when doing the analysis!**

We use semantic versioning (http://semver.org), and the current version of
ExpAn is: v0.2.4.

The version is maintained in ``setup.cfg``, and propagated from there to various files
by the ``bumpversion`` program. The most important propagation destination is
in ``version.py`` where it is held in the string ``__version__`` with
the form:

::

    '{major}.{minor}.{patch}'

The ``__version__`` string and a ``version()`` function is imported by
``core.__init__`` and so is accessible to imported functions in expan.

The ``version(format_str)`` function generates version strings of any
form. It can use git's commit count and revision number to generate a
long version string which may be useful for pip versioning? Examples:
NB: caution using this... it won't work if not in the original git
repository.

::

    >>> import core.binning
    >>> core.version()
    'v0.2.4'
    >>> core.version('{major}.{minor}..{commits}')
    '0.0..176'
    >>> core.version('{commit}')
    'a24730a42a4b5ae01bbdb05f6556dedd453c1767'

See: `StackExchange
151558 <http://programmers.stackexchange.com/a/151558>`__

Bumping Version
---------------

Can use bumpversion to maintain the ``__version__`` in ``version.py``:

::

    $ bumpversion patch
    or
    $ bumpversion minor

This will update the version number, create a new tag in git, and commit
the changes with a standard commit message.

When you have done this, you must push the commit and new tag to the
repository with:

::

    $ git push --tags

Travis CI and PyPI deployment
-----------------------------

We use Travis CI for testing builds and deploying our PyPI package. Currently, a build and test is triggered when a pull request or push to 'dev' or 'master' is made. When a tagged commit to master is made then a build, test and deploy is made.

If you wish to skip triggering a CI task, please include "[ci skip]" in your commit message.

When a tagged commit to master is made this triggers a build and this build will also deploy the created package to PyPI.

TODO
----

-  parallelization, eg. for the bootstrapping code
-  Bayesian updating/early stopping
-  multiple comparison correction, definitely relevant for delta and
   SGA, have to think about how to correct for time dependency in the
   trend analysis
-  implement from\_json and to\_json methods in the Binning class, in
   order to convert the Python object to a json format for persisting in
   the Results metadata and reloading from a script

License
=======

The MIT License (MIT)

Copyright © [2016] Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


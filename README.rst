==========================
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

.. image:: https://img.shields.io/pypi/l/expan.svg
        :target: https://pypi.python.org/pypi/expan/
        :alt: License

.. image:: https://readthedocs.org/projects/expan/badge/?version=latest
        :target: http://expan.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

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


Installation
============

To install ExpAn, run this command in your terminal:

.. code-block:: console

    $ pip install expan


Usage
=====

To use ExpAn in a project::

    import expan

Some mock-up data:

::

    from expan.core.experiment import Experiment
    from tests.tests_core.test_data import generate_random_data

    exp = Experiment('B', *generate_random_data())
    exp.delta()



Documentation
=============

The latest stable version is 0.4.4.

`ExpAn main documentation <http://expan.readthedocs.io/>`__

`ExpAn Description <https://github.com/zalando/expan/blob/master/ExpAn-Description.mediawiki>`__ - details about the concept of the library and data structures.

`ExpAn Introduction <https://github.com/zalando/expan/blob/dev/ExpAn-Intro.ipynb>`__ - a full jupyter (iPython) notebook. You can view it as slides with `jupyter <http://jupyter.org>`__:

::

    sh serve_intro_slides


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

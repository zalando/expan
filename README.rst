==========================
ExpAn: Experiment Analysis
==========================

.. image:: https://img.shields.io/travis/zalando/expan.svg
        :target: https://travis-ci.org/zalando/expan
        :alt: Build status

.. image:: https://coveralls.io/repos/github/zalando/expan/badge.svg
        :target: https://coveralls.io/github/zalando/expan
        :alt: Code coverage

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


Documentation
=============

The latest stable version is 1.4.0. Please check out our `tutorial and documentation <http://expan.readthedocs.io/>`__.

Installation
============

Stable release
--------------

To install ExpAn, run this command in your terminal:

.. code-block:: console

    $ pip install expan

From sources
------------

The sources for ExpAn can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/zalando/expan

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/zalando/expan/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/zalando/expan
.. _tarball: https://github.com/zalando/expan/tarball/master


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

.. highlight:: shell

============
Contributing
============

Style guide
===========

We follow `PEP8 standards <https://www.python.org/dev/peps/pep-0008>`__
with the following exceptions:

- Use *tabs instead of spaces* - this allows all individuals to have visual depth of indentation they prefer, without changing the source code at all, and it is simply smaller

Testing
=======

Easiest way to run tests is by running the command ``tox`` from the terminal. The default Python environments for testing with are py27 and py34, but you can specify your own by running e.g. ``tox -e py35``.

Branching / Release
===================

We currently use the gitflow workflow. Feature branches are created from
and merged back to the ``dev`` branch, and the ``master`` branch stores
snapshots/releases of the ``dev`` branch.

See also the much simpler github flow
`here <http://scottchacon.com/2011/08/31/github-flow.html>`__

Versioning
==========

**For the sake of reproducibility, always be sure to work with a release
when doing the analysis!**

We use semantic versioning (http://semver.org), and the current version of
ExpAn is: v0.4.0.

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
    'v0.4.0'
    >>> core.version('{major}.{minor}..{commits}')
    '0.0..176'
    >>> core.version('{commit}')
    'a24730a42a4b5ae01bbdb05f6556dedd453c1767'

See: `StackExchange
151558 <http://programmers.stackexchange.com/a/151558>`__

Bumping Version
===============

Can use bumpversion to maintain the ``__version__`` in ``version.py``:

::

    $ bumpversion patch

or

::

    $ bumpversion minor

This will update the version number, create a new tag in git, and commit
the changes with a standard commit message.

When you have done this, you must push the commit and new tag to the
repository with:

::

    $ git push --tags

Travis CI and PyPI deployment
=============================

We use Travis CI for testing builds and deploying our PyPI package.

A **build** and **test** is triggered when a commit is pushed to either

- **dev**,
- **master**
- or a **pull request branch to dev or master**.

If you want to **deploy to PyPI**, then follow these steps:

- assuming you have a dev branch that is up to date, create a pull request from dev to master (a travis job will be started for the pull request)
- once the pull request is approved, merge it (another travis job will be started because a push to master happened)
- checkout master
- push **tags** to **master** (a third travis job will be started, but this time it will also push to PyPI because tags were pushed)

If you wish to skip triggering a CI task (for example when you change documentation), please include ``[ci skip]`` in your commit message.

TODOs
=====

- parallelization, eg. for the bootstrapping code
- Bayesian updating/early stopping
- multiple comparison correction, definitely relevant for delta and SGA, have to think about how to correct for time dependency in the trend analysis
- implement from\_json and to\_json methods in the Binning class, in order to convert the Python object to a json format for persisting in the Results metadata and reloading from a script

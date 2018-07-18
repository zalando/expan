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
=========

Easiest way to run tests is by running the command ``tox`` from the terminal. The default Python environments for testing are python 2.7 and python 3.6.
You can also specify your own by running e.g. ``tox -e py35``.

Branching
===================

We currently use the gitflow workflow. Feature branches are created from
and merged back to the ``master`` branch. Please always make a Pull Request
when you contribute.

See also the much simpler github flow
`here <http://scottchacon.com/2011/08/31/github-flow.html>`__


Release
=================

To make a release and **deploy to PyPI**, please follow these steps (we highly suggest to leave the release to admins of ExpAn):

- assuming you have a master branch that is up to date, create a pull request from your feature branch to master (a travis job will be started for the pull request)
- once the pull request is approved, merge it (another travis job will be started because a push to master happened)
- checkout master
- create a new tag
- run documentation generation which includes creation of changelog
- push **tags** to **master** (a third travis job will be started, but this time it will also push to PyPI because tags were pushed)

The flow would then look like follows:

1. :code:`bumpversion patch` or :code:`bumpversion minor`
2. :code:`git describe --tags`, and note this tag name. We will need to edit this tag later.
3. :code:`make docs`, which will extend the changelog by reading information from github.com/zalando/expan .
4. :code:`git add CHANGELOG.*`
5. :code:`git commit --amend --no-edit`
6. :code:`git show`. Carefully review this commit before proceeding. Ensure the changelog is updated with the expected text, in particular a fully up-to-date version number.
7. :code:`git tag -d {TAGNAME}`, where :code:`{TAGNAME}` is the tag name from step 2.
8. :code:`git tag    {TAGNAME}` to recreate the tag in the correct place.
9. :code:`git push`
10. :code:`git push --tags`

You can then check if the triggered Travis CI job is tagged (the name should be eg. 'v1.2.3' instead of 'master').

Note that this workflow has a flaw that changelog generator will not put the changes of the current release,
because it reads the commit messages from git remote.

Solution: We need to run ``make docs`` on **master** once more *after the release* to update the documentation page.

A better solution could be to discard the automatic changelog generator and manually write the changelog before step 1,
and then config ``make docs`` to use this changelog file.


We explain the individual steps below.


Sphinx documentation
-----------------------
:code:`make docs` will create the html documentation if you have sphinx installed.
You might need to install our theme explicitly by :code:`pip install sphinx_rtd_theme`.

If you have encountered an error like this:
:code:`API rate limit exceeded for github_username`, you need to create a git token and set an environment variable for it.
See instructions `here <https://github.com/skywinder/github-changelog-generator#github-token>`__.



Versioning
----------------

For the sake of reproducibility, always be sure to work with a release
when doing the analysis. We use `semantic versioning <http://semver.org>`__.

The version is maintained in ``setup.cfg``, and propagated from there to various files
by the ``bumpversion`` program. The most important propagation destination is
in ``version.py`` where it is held in the string ``__version__`` with
the form:

::

    '{major}.{minor}.{patch}'


Bumping Version
----------------

We use bumpversion to maintain the ``__version__`` in ``version.py``:

::

    $ bumpversion patch

or

::

    $ bumpversion minor

This will update the version number, create a new tag in git, and commit
the changes with a standard commit message.


Travis CI
----------------

We use Travis CI for testing builds and deploying our PyPI package.

A **build** with unit tests is triggered either

- a commit is pushed to **master**
- or a **pull request** to **master** is opened.

A release to PyPI will be triggered if a new tag is pushed to **master**.

If you wish to skip triggering a CI task (for example when you change documentation), please include ``[ci skip]`` in your commit message.

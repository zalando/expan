"""ExpAn core module.
"""

from __future__ import absolute_import

from expan.core.version import __version__, version

__all__ = ["binning", "early_stopping", "experiment", "statistics", "util",
           "version", "results", "correction", "statistical_test"]

print(('ExpAn core init: {}'.format(version())))

"""ExpAn core module.
"""

from __future__ import absolute_import

# __all__ = ["binning", "experiment", "experimentdata", "results", "statistics", "util", "version"]
__all__ = ["binning", "experiment", "statistics", "util", "version"]

from expan.core.version import __version__, version

print(('ExpAn core init: {}'.format(version())))

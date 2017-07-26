"""Main ExpAn module that contains the cli, core and data modules.
"""

from __future__ import absolute_import

import logging.config

# from expan.cli import *
from expan.core import *
from expan.core.version import __version__
from expan.data import *

__all__ = ["core", "data", "cli"]
logging.basicConfig(level=logging.DEBUG)

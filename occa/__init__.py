import os
from os.path import abspath, dirname

from ._version import get_versions


if 'OCCA_DIR' not in os.environ:
    os.environ['OCCA_DIR'] = abspath(
        os.path.join(dirname(__file__), 'c', 'occa.git')
    )


__version__ = get_versions()['version']
del get_versions


from .base import *
from .device import Device
from .memory import Memory
from .kernel import Kernel
from .stream import Stream
from .streamtag import StreamTag
from . import okl

import os
import sys
from os.path import abspath, dirname

from ._version import get_versions


# Setup version
__version__ = get_versions()['version']
del get_versions


# Setup OCCA_DIR
OCCA_DIR = os.environ.get('OCCA_DIR')
if not OCCA_DIR:
    OCCA_DIR = abspath(os.path.join(dirname(__file__), 'c', 'occa.git'))
    os.environ['OCCA_DIR'] = OCCA_DIR


# Package imports
from .base import *
from .device import Device
from .memory import Memory
from .kernel import Kernel
from .stream import Stream
from .streamtag import StreamTag
from . import okl

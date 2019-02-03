from glob import glob
import os
import sys
from os.path import abspath, dirname

from ._version import get_versions
from .exceptions import Error


# Setup version
__version__ = get_versions()['version']
del get_versions


# Setup OCCA_DIR
PY_OCCA_DIR = abspath(os.path.join(dirname(__file__), 'c', 'occa.git'))
OCCA_DIR = os.environ.get('OCCA_DIR')
if OCCA_DIR and OCCA_DIR != PY_OCCA_DIR:
    raise Error('Environment variable [OCCA_DIR] should not be set')
os.environ['OCCA_DIR'] = PY_OCCA_DIR


# Check LD_LIBRARY_PATH
LD_LIBRARY_PATH_VAR = (
    'LD_LIBRARY_PATH'
    if sys.platform != 'darwin'
    else 'DYLD_LIBRARY_PATH'
)
LD_LIBRARY_PATH = os.environ.get(LD_LIBRARY_PATH_VAR) or ''
PY_OCCA_DIR_LIB = os.path.join(PY_OCCA_DIR, 'lib')
for entry in LD_LIBRARY_PATH.split(':'):
    if entry == PY_OCCA_DIR_LIB:
        continue
    if glob('{path}/libocca.so*'.format(path=entry)):
        raise Error('libocca.so should not be in [{path_env}]'.format(
            path_env=LD_LIBRARY_PATH_VAR
        ))


# Package imports
from .base import *
from .device import Device
from .memory import Memory
from .kernel import Kernel
from .stream import Stream
from .streamtag import StreamTag
from . import okl

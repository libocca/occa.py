import collections
import json
import numpy as np
import warnings

from . import c, exceptions


PY_TO_DTYPE = {
    bool: np.dtype(bool),
    int: np.dtype(int),
    float: np.dtype(float),
}


VALID_PY_TYPES = {
    bool,
    int,
    float,
}


VALID_NP_TYPES = {
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
}


VALID_NP_DTYPES = {
    np.dtype(t)
    for t in VALID_NP_TYPES
}

VALID_TYPES = set(PY_TO_DTYPE.keys())
VALID_TYPES.update(VALID_NP_DTYPES)


TYPES_TO_C_TYPES = {
    type(None): 'void',
    bool: 'bool',
    int: 'long',
    float: 'double',
    str: 'char *',
    np.bool_: 'bool',
    np.int8: 'char',
    np.uint8: 'char',
    np.int16: 'short',
    np.uint16: 'short',
    np.int32: 'int',
    np.uint32: 'int',
    np.int64: 'long',
    np.uint64: 'long',
    np.float32: 'float',
    np.float64: 'double',
}


def is_int(value):
    return (isinstance(value, int) or
            isinstance(value, np.integer))


#---[ Assert C ]------------------------
def assert_c_device(value):
    if not isinstance(value, c.Device):
        raise ValueError('Expected an occa.c.Device')


def assert_c_memory(value):
    if not isinstance(value, c.Memory):
        raise ValueError('Expected an occa.c.Memory')


def assert_c_kernel(value):
    if not isinstance(value, c.Kernel):
        raise ValueError('Expected an occa.c.Kernel')


def assert_c_stream(value):
    if not isinstance(value, c.Stream):
        raise ValueError('Expected an occa.c.Stream')


def assert_c_streamtag(value):
    if not isinstance(value, c.StreamTag):
        raise ValueError('Expected an occa.c.StreamTag')
#=======================================


#---[ Assert Py ]-----------------------
def assert_str(value):
    if not isinstance(value, str):
        raise ValueError('Expected an str')


def assert_int(value):
    if not is_int(value):
        raise ValueError('Expected an int')


def assert_device(value):
    from .device import Device

    if not isinstance(value, Device):
        raise ValueError('Expected an occa.Device')


def assert_memory(value):
    from .memory import Memory
    if not isinstance(value, Memory):
        raise ValueError('Expected an occa.Memory')


def assert_kernel(value):
    from .kernel import Kernel
    if not isinstance(value, Kernel):
        raise ValueError('Expected an occa.Kernel')


def assert_stream(value):
    from .stream import Stream
    if not isinstance(value, Stream):
        raise ValueError('Expected an occa.Stream')


def assert_streamtag(value):
    from .streamtag import StreamTag
    if not isinstance(value, StreamTag):
        raise ValueError('Expected an occa.StreamTag')


def assert_properties(value, **kwargs):
    if not (value is None or
            isinstance(value, str) or
            isinstance(value, dict)):
        raise ValueError('Props is expected to be None, str, or dict')

    if len(kwargs) > 0:
        try:
            json.dumps(kwargs)
        except Exception as e:
            raise type(e)(
                '**kwargs are not json serializable: {}'.format(e)
            )


def assert_dim(value):
    if (not isinstance(value, collections.Iterable) or
        len(list(value)) > 3):
        raise ValueError('Expected an iterable of at most size 3')


def assert_valid_dtype(value):
    if value not in VALID_TYPES:
        raise ValueError('Type [{value}] is not a valid type: {types}'.format(
            value=value,
            types=sorted([
                (('np.' + t.__name__)
                 if t in VALID_NP_DTYPES
                 else t.__name__)
                for t in VALID_TYPES
            ]),
        ))


def assert_ndarray(value):
    if not isinstance(value, np.ndarray):
        raise ValueError('Expected a numpy.ndarray')


def assert_malloc_src(value):
    if (not is_int(value) and
        not isinstance(value, np.ndarray) and
        not isinstance(value, list)):
        raise ValueError('Expected an int, numpy.ndarray, or list')


def assert_memory_like(value):
    from .memory import Memory

    if (not isinstance(value, Memory) and
        not isinstance(value, np.ndarray) and
        not isinstance(value, list)):
        raise ValueError('Expected occa.Memory, numpy.ndarray, or list')
#=======================================


#---[ Type Conversions ]----------------
def properties(props, **kwargs):
    if not (props is None or
            isinstance(props, str) or
            isinstance(props, dict)):
        raise ValueError('Props is expected to be None, str, or dict')

    has_props = props is not None
    has_kwargs = len(kwargs) > 0

    if has_props:
        if has_kwargs:
            props.update(kwargs)
            return json.dumps(props)
        # No kwargs
        if isinstance(props, dict):
            props = json.dumps(props)
        return props

    # Only kwargs
    if has_kwargs:
        return json.dumps(kwargs)

    # Nothing
    return None


def cast_arg(value):
    # Memory and custom types
    if hasattr(value, '_to_occa_kernel_arg'):
        return value._to_occa_kernel_arg()

    valtype = type(value)

    # Pod
    to_dtype = PY_TO_DTYPE.get(valtype)
    if to_dtype:
        return to_dtype(value)

    # Numpy dtype or None
    if (valtype in VALID_NP_TYPES or
        value is None):
        return value

    raise TypeError('Unsupported type for a kernel argument: [{argtype}]'
                    .format(argtype=type(value)))


def cast_args(args):
    return [cast_arg(arg) for arg in args]


def memory_buffer(value,
                  entries=None,
                  buffer=None):
    assert_memory_like(value)

    if isinstance(value, np.ndarray):
        return None

    entries = entries or len(value)

    # Test buffer
    if buffer is not None:
        assert_ndarray(buffer)
        if len(buffer) < entries:
            warnings.warn('Buffer is too small, ignoring buffer',
                          exceptions.BufferWarning)
            buffer = None
        elif buffer.dtype != value.dtype:
            warnings.warn('Buffer dtype differs from value, ignoring buffer',
                          exceptions.BufferWarning)
            buffer = None
        else:
            return buffer

    return np.zeros(entries,
                    dtype=value.dtype)
#=======================================

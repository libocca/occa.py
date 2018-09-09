#
# The MIT License (MIT)
#
# Copyright (c) 2018 David Medina
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#
import collections
import json
import numpy as np

from . import c


PY_TO_DTYPE = {
    int: np.dtype(int).type,
    float: np.dtype(float).type,
    bool: np.dtype(bool).type,
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


def assert_c_streamTag(value):
    if not isinstance(value, c.StreamTag):
        raise ValueError('Expected an occa.c.StreamTag')
#=======================================

#---[ Assert Py ]-----------------------
def assert_str(value):
    if not isinstance(value, str):
        raise ValueError('Expected an str')


def assert_int(value):
    if not isinstance(value, int):
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


def assert_streamTag(value):
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


def assert_ndarray(value):
    if not isinstance(value, np.ndarray):
        raise ValueError('Expected a numpy.ndarray')


def assert_memory_like(value):
    from .memory import Memory

    if (not isinstance(value, Memory) and
        not isinstance(value, np.ndarray)):
        raise ValueError('Expected occa.Memory or numpy.ndarray')

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
    return [
        cast_arg(arg)
        for arg in args
    ]
#=======================================

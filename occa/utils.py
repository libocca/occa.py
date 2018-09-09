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

from . import c, device, memory, kernel, stream, streamtag


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
def assert_c_device(d):
    if not isinstance(d, c.Device):
        raise ValueError('Expected an occa.c.Device')


def assert_c_memory(d):
    if not isinstance(d, c.Memory):
        raise ValueError('Expected an occa.c.Memory')


def assert_c_kernel(d):
    if not isinstance(d, c.Kernel):
        raise ValueError('Expected an occa.c.Kernel')


def assert_c_stream(d):
    if not isinstance(d, c.Stream):
        raise ValueError('Expected an occa.c.Stream')


def assert_c_streamTag(d):
    if not isinstance(d, c.StreamTag):
        raise ValueError('Expected an occa.c.StreamTag')
#=======================================

#---[ Assert Py ]-----------------------
def assert_str(d):
    if not isinstance(d, str):
        raise ValueError('Expected an str')


def assert_int(d):
    if not isinstance(d, int):
        raise ValueError('Expected an int')


def assert_device(d):
    if not isinstance(d, Device):
        raise ValueError('Expected an occa.Device')


def assert_memory(d):
    if not isinstance(d, Memory):
        raise ValueError('Expected an occa.Memory')


def assert_kernel(d):
    if not isinstance(d, Kernel):
        raise ValueError('Expected an occa.Kernel')


def assert_stream(d):
    if not isinstance(d, Stream):
        raise ValueError('Expected an occa.Stream')


def assert_streamTag(d):
    if not isinstance(d, StreamTag):
        raise ValueError('Expected an occa.StreamTag')


def assert_properties(props, **kwargs):
    if not (props is None or
            isinstance(props, str) or
            isinstance(props, dict)):
        raise ValueError('Props is expected to be None, str, or dict')

    if len(kwargs) > 0:
        try:
            json.dumps(kwargs)
        except Exception as e:
            raise type(e)(
                '**kwargs are not json serializable: {}'.format(e)
            )


def assert_dim(d):
    if (not isinstance(d, collections.Iterable) or
        len(list(d)) > 3):
        raise ValueError('Expected an iterable of at most size 3')
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
            return json.dumps({
                **props
                **kwargs
            })
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
        return PackedArg.to_dtype(to_dtype(value))

    # Numpy dtype/ndarray or None
    if (valtype in VALID_NP_TYPES or
        isinstance(value, np.ndarray) or
        value is None):
        return value

    raise TypeError('Unsupported type for a kernel argument: [{argtype}]'
                    .format(argtype=type(value)))


def args(args):
    return [
        cast_art(arg)
        for arg in args
    ]
#=======================================

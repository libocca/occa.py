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

from . import c, device, memory, kernel, stream, streamtag


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
        len(list(d)) != 3):
        raise ValueError('Expected an iterable of size 3')
#=======================================


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

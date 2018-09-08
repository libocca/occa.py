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
import json

from . import device, memory, kernel, stream, streamtag


def assert_device(d):
    if not isinstance(d, device.Device):
        raise ValueError('Expected an occa.Device')


def assert_memory(d):
    if not isinstance(d, memory.Memory):
        raise ValueError('Expected an occa.Memory')


def assert_kernel(d):
    if not isinstance(d, kernel.Kernel):
        raise ValueError('Expected an occa.Kernel')


def assert_stream(d):
    if not isinstance(d, stream.Stream):
        raise ValueError('Expected an occa.Stream')


def assert_streamTag(d):
    if not isinstance(d, streamTag.StreamTag):
        raise ValueError('Expected an occa.StreamTag')


def assert_properties(props, **kwargs):
    if not (props is None or
            isinstance(prop, str) or
            isinstance(props, dict)):
        raise ValueError('Props is expected to be None, str, or dict')

    if len(kwargs) > 0:
        try:
            json.dumps(kwargs)
        except Exception as e:
            raise type(e)(
                '**kwargs are not json serializable: {}'.format(e)
            )


def properties(props, **kwargs):
    if not (props is None or
            isinstance(prop, str) or
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

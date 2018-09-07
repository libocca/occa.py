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
import functools
import json

from . import c
from .exceptions import UninitializedError


class Device(c.Device):
    def __init__(self, props=None, **kwargs):
        if not (props is None or
                isinstance(prop, str) or
                isinstance(props, dict)):
            raise ValueError('Expected a str, dict, or properties through **kwargs')

        if isinstance(props, str):
            props = json.loads(props)

        # Override props if kwargs are defined
        if len(kwargs):
            if props is None:
                props = kwargs
            else:
                props = {
                    **props,
                    **kwargs,
                }

        if props is None:
            return super().__init__()
        return super().__init__(props=props)

    @property
    def _c(self):
        return super(Device, self)

    def _assert_initialized(self):
        if not self.is_initialized():
            raise UninitializedError('occa.Device')

    def is_initialized(self):
        '''Return if the device has been initialized'''
        return self._c.is_initialized()

    def free(self):
        self._assert_initialized()
        c.device.free(self._handle)

    @property
    def mode(self):
        self._assert_initialized()
        return c.device.mode(self._handle)

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(c.device.properties(self._handle))

    @property
    def kernel_properties(self):
        self._assert_initialized()
        return json.loads(c.device.kernel_properties(self._handle))

    @property
    def memory_properties(self):
        self._assert_initialized()
        return json.loads(c.device.memory_properties(self._handle))

    def memory_size(self):
        self._assert_initialized()
        return c.device.memory_size(self._handle)

    def memory_allocated(self):
        self._assert_initialized()
        return c.device.memory_allocated(self._handle)

    def finish(self):
        self._assert_initialized()
        c.device.finish(self._handle)

    @property
    def has_separate_memory_space(self):
        self._assert_initialized()
        return c.device.has_separate_memory_space(self._handle)

    #---[ Stream ]----------------------
    def create_stream(self):
        self._assert_initialized()
        return c.device.create_stream(self._handle)

    @property
    def stream(self):
        self._assert_initialized()
        return c.device.stream(self._handle)

    def set_stream(self, stream):
        self._assert_initialized()
        if not isinstance(stream, Stream):
            raise ValueError('Expected occa.Stream')
        return c.device.set_stream(self._handle,
                                   stream._stream)

    def free_stream(self, stream):
        self._assert_initialized()
        if not isinstance(stream, Stream):
            raise ValueError('Expected occa.Stream')
        c.device.free_stream(self._handle,
                             stream._stream)

    def tag_stream(self):
        self._assert_initialized()
        return c.device.tag_stream(self._handle)

    def wait_for_tag(self, tag):
        self._assert_initialized()
        if not isinstance(tag, Tag):
            raise ValueError('Expected occa.Tag')
        c.device.wait_for_tag(self._handle,
                              tag._tag)

    def time_between_tags(self, start_tag, end_tag):
        self._assert_initialized()
        if (not isinstance(start_tag, Tag) or
            not isinstance(end_tag, Tag)):
            raise ValueError('Expected occa.Tag')

        return c.device.time_between_tags(self._handle,
                                          start_tag._tag,
                                          end_tag._tag)
    #===================================

    #---[ Kernel ]----------------------
    def build_kernel(self, filename, kernel_name, props):
        self._assert_initialized()
        return c.device.build_kernel(self._handle,
                                     filename,
                                     kernel_name,
                                     json.dumps(props))

    def build_kernel_from_string(self, source, kernel_name, props):
        self._assert_initialized()
        return c.device.build_kernel_from_string(self._handle,
                                                 source,
                                                 kernel_name,
                                                 json.dumps(props))
    #===================================

    #---[ Memory ]----------------------
    def malloc(self, bytes, src, props):
        self._assert_initialized()
        return c.malloc(self._handle,
                        bytes,
                        src,
                        props)
    #===================================

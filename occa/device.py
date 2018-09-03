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
from . import c
from .stream import Stream
from .tag import Tag
from .utils import defaults_to


class Device:
    def __init__(self, props=None, **kwargs):
        self._handle = None

        if props is None:
            props = kwargs
        if isinstance(props, dict):
            props = json.dumps(props)
        if props is None:
            return

        if isinstance(props, str):
            self._handle = c.device.create(props)
        else:
            raise ValueError('Expected a str, dict, or **kwargs')

    @defaults_to(False)
    def is_initialized(self):
        return True

    @defaults_to(None)
    def free(self):
        c.device.free(self._handle)

    @property
    @defaults_to('')
    def mode(self):
        return c.device.mode(self._handle)

    @property
    @defaults_to({})
    def properties(self):
        return json.loads(c.device.properties(self._handle))

    @property
    @defaults_to({})
    def kernel_properties(self):
        return json.loads(c.device.kernel_properties(self._handle))

    @property
    @defaults_to({})
    def memory_properties(self):
        return json.loads(c.device.memory_properties(self._handle))

    @defaults_to(0)
    def memory_size(self):
        return c.device.memory_size(self._handle)

    @defaults_to(0)
    def memory_allocated(self):
        return c.device.memory_allocated(self._handle)

    @defaults_to(None)
    def finish(self):
        c.device.finish(self._handle)

    @property
    @defaults_to(False)
    def has_separate_memory_space(self):
        return c.device.has_separate_memory_space(self._handle)

    #---[ Stream ]----------------------
    @defaults_to(None)
    def create_stream(self):
        return c.device.create_stream(self._handle)

    @property
    @defaults_to(None)
    def stream(self):
        return c.device.stream(self._handle)

    @defaults_to(None)
    def set_stream(self, stream):
        if not isinstance(stream, Stream):
            raise ValueError('Expected occa.Stream')
        return c.device.set_stream(self._handle,
                                   stream._stream)

    @defaults_to(None)
    def free_stream(self, stream):
        if not isinstance(stream, Stream):
            raise ValueError('Expected occa.Stream')
        c.device.free_stream(self._handle,
                             stream._stream)

    @defaults_to(None)
    def tag_stream(self):
        return c.device.tag_stream(self._handle)

    @defaults_to(None)
    def wait_for_tag(self, tag):
        if not isinstance(tag, Tag):
            raise ValueError('Expected occa.Tag')
        c.device.wait_for_tag(self._handle,
                              tag._tag)

    @defaults_to(0)
    def time_between_tags(self, start_tag, end_tag):
        if (not isinstance(start_tag, Tag) or
            not isinstance(end_tag, Tag)):
            raise ValueError('Expected occa.Tag')

        return c.device.time_between_tags(self._handle,
                                          start_tag._tag,
                                          end_tag._tag)
    #===================================

    #---[ Kernel ]----------------------
    @defaults_to(None)
    def build_kernel(self, filename, kernel_name, props):
        return c.device.build_kernel(self._handle,
                                     filename,
                                     kernel_name,
                                     json.dumps(props))

    @defaults_to(None)
    def build_kernel_from_string(self, source, kernel_name, props):
        return c.device.build_kernel_from_string(self._handle,
                                                 source,
                                                 kernel_name,
                                                 json.dumps(props))
    #===================================

    #---[ Memory ]----------------------
    @defaults_to(None)
    def malloc(self, bytes, src, props):
        return c.malloc(self._handle,
                        bytes,
                        src,
                        props)

    @defaults_to(None)
    def umalloc(self, bytes, src, props):
        return c.umalloc(self._handle,
                         bytes,
                         src,
                         props)
    #===================================

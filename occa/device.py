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

from . import c, utils
from .exceptions import UninitializedError


class Device:
    def __init__(self, props=None, **kwargs):
        if isinstance(props, c.Device):
            self._c = props
            return

        utils.assert_properties(props, **kwargs)
        props = utils.properties(props, **kwargs)

        if props is not None:
            self._c = c.Device(props=props)
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.Device is not initialized')

    @property
    def is_initialized(self):
        '''Return if the device has been initialized'''
        return self._c and self._c.is_initialized()

    def free(self):
        self._assert_initialized()
        self._c.free()

    @property
    def mode(self):
        self._assert_initialized()
        return self._c.mode()

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(self._c.properties())

    @property
    def kernel_properties(self):
        self._assert_initialized()
        return json.loads(self._c.kernel_properties())

    @property
    def memory_properties(self):
        self._assert_initialized()
        return json.loads(self._c.memory_properties())

    @property
    def memory_size(self):
        self._assert_initialized()
        return self._c.memory_size()

    @property
    def memory_allocated(self):
        self._assert_initialized()
        return self._c.memory_allocated()

    def finish(self):
        self._assert_initialized()
        self._c.finish()

    @property
    def has_separate_memory_space(self):
        self._assert_initialized()
        return self._c.has_separate_memory_space()

    #---[ Stream ]----------------------
    def create_stream(self):
        from .stream import Stream

        self._assert_initialized()
        return Stream(self._c.create_stream())

    @property
    def stream(self):
        from .stream import Stream

        self._assert_initialized()
        return Stream(self._c.get_stream())

    def set_stream(self, stream):
        self._assert_initialized()
        utils.assert_stream(stream)

        self._c.set_stream(stream._c)

    def tag_stream(self):
        from .streamtag import StreamTag

        self._assert_initialized()

        return StreamTag(self._c.tag_stream())

    def wait_for(self, tag):
        self._assert_initialized()
        utils.assert_streamtag(tag)

        self._c.wait_for(tag._c)

    def time_between(self, start, end):
        self._assert_initialized()
        utils.assert_streamtag(start)
        utils.assert_streamtag(end)

        return self._c.time_between(start._c, end._c)
    #===================================

    #---[ Kernel ]----------------------
    def build_kernel(self, filename, kernel, props=None):
        from .kernel import Kernel

        self._assert_initialized()
        utils.assert_str(filename)
        utils.assert_str(kernel)
        props = utils.properties(props) or ''

        return Kernel(
            self._c.build_kernel(filename=filename,
                                 kernel=kernel,
                                 props=props)
        )

    def build_kernel_from_string(self, source, kernel, props=None):
        from .kernel import Kernel

        self._assert_initialized()
        utils.assert_str(source)
        utils.assert_str(kernel)
        props = utils.properties(props) or ''

        return Kernel(
            self._c.build_kernel_from_string(source=source,
                                             kernel=kernel,
                                             props=props)
        )
    #===================================

    #---[ Memory ]----------------------
    def malloc(self, bytes=None, src=None, props=None):
        from .memory import Memory

        self._assert_initialized()
        if bytes is not None:
            utils.assert_int(bytes)
        if src is not None:
            utils.assert_ndarray(src)
        props = utils.properties(props)

        return Memory(
            self._c.malloc(bytes=bytes,
                           src=src,
                           props=props)
        )
    #===================================

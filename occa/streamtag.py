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

from . import c, utils
from .exceptions import UninitializedError


class StreamTag:
    def __init__(self, c_streamtag=None):
        if c_streamtag:
            utils.assert_c_stream_tag(c_streamtag)
            self._c = c_streamtag
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.StreamTag is not initialized')

    @property
    def is_initialized(self):
        '''Return if the stream tag has been initialized'''
        return self._c.is_initialized()

    def free(self):
        self._assert_initialized()
        self._c.free()

    @property
    def device(self):
        from .device import Device

        self._assert_initialized()
        return Device(self._c.get_device())

    def wait(self):
        self._assert_initialized()
        self._c.wait()

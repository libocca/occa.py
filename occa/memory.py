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
from .exceptions import UninitializedError


class Memory(c.Memory):
    def __init__(self):
        pass

    @property
    def _c(self):
        return super(Memory, self)

    def _assert_initialized(self):
        if not self.is_initialized():
            raise UninitializedError('occa.Memory')

    def is_initialized(self):
        '''Return if the memory has been initialized'''
        return self._c.is_initialized()

    def free(self):
        self._assert_initialized()
        c.memory.free(self._handle)

    @property
    def device(self):
        self._assert_initialized()
        return c.memory.device(self._handle)

    @property
    def mode(self):
        self._assert_initialized()
        return c.memory.mode(self._handle)

    @property
    def size(self):
        self._assert_initialized()
        return c.memory.size(self._handle)

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(c.memory.properties(self._handle))

    def copy_to(self, dest, bytes, offset, props):
        self._assert_initialized()
        c.memory.copy_to(self._handle,
                         dest._handle,
                         bytes,
                         offset,
                         json.dumps(props))

    def copy_from(self, src, bytes, offset, props):
        self._assert_initialized()
        c.memory.copy_from(self._handle,
                         src._handle,
                         bytes,
                         offset,
                         json.dumps(props))

    def clone(self):
        self._assert_initialized()
        return c.memory.clone(self._handle)

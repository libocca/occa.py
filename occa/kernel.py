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


class Kernel(c.Kernel):
    def __init__(self):
        pass

    @property
    def _c(self):
        return super(Kernel, self)

    def _assert_initialized(self):
        if not self.is_initialized():
            raise UninitializedError('occa.Kernel')

    def is_initialized(self):
        '''Return if the kernel has been initialized'''
        return self._c.is_initialized()

    def free(self):
        self._assert_initialized()
        c.kernel.free(self._handle)

    @property
    def device(self):
        self._assert_initialized()
        return c.kernel.device(self._handle)

    @property
    def mode(self):
        self._assert_initialized()
        return c.kernel.mode(self._handle)

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(c.kernel.properties(self._handle))

    @property
    def name(self):
        self._assert_initialized()
        return c.kernel.name(self._handle)

    @property
    def source_filename(self):
        self._assert_initialized()
        return c.kernel.source_filename(self._handle)

    @property
    def binary_filename(self):
        self._assert_initialized()
        return c.kernel.binary_filename(self._handle)

    @property
    def max_dims(self):
        self._assert_initialized()
        return c.kernel.max_dims(self._handle)

    @property
    def max_outer_dims(self):
        self._assert_initialized()
        return c.kernel.max_outer_dims(self._handle)

    @property
    def max_inner_dims(self):
        self._assert_initialized()
        return c.kernel.max_inner_dims(self._handle)

    def set_run_dims(self, outer_dims, inner_dims):
        self._assert_initialized()
        c.kernel.set_run_dims(self._handle,
                              outer_dims,
                              inner_dims)

    def __call__(self):
        self._assert_initialized()
        pass

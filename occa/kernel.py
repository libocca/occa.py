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


class Kernel:
    def __init__(self, c_kernel=None):
        if c_kernel:
            utils.assert_c_kernel(c_kernel)
            self._c = c_kernel
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.Kernel is not initialized')

    @property
    def is_initialized(self):
        '''Return if the kernel has been initialized'''
        return self._c.is_initialized()

    def free(self):
        self._assert_initialized()
        self._c.free()

    @property
    def device(self):
        from .device import Device

        self._assert_initialized()
        return Device(self._c.get_device())

    @property
    def mode(self):
        self._assert_initialized()
        return self._c.mode()

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(self._c.properties())

    @property
    def name(self):
        self._assert_initialized()
        return self._c.name()

    @property
    def source_filename(self):
        self._assert_initialized()
        return self._c.source_filename()

    @property
    def binary_filename(self):
        self._assert_initialized()
        return self._c.binary_filename()

    @property
    def max_dims(self):
        self._assert_initialized()
        return self._c.max_dims()

    @property
    def max_outer_dims(self):
        self._assert_initialized()
        return self._c.max_outer_dims()

    @property
    def max_inner_dims(self):
        self._assert_initialized()
        return self._c.max_inner_dims()

    def set_run_dims(self, outer, inner):
        self._assert_initialized()
        utils.assert_dim(outer)
        utils.assert_dim(inner)

        self._c.set_run_dims(outer=list(outer),
                             inner=list(inner))

    def __call__(self, *args):
        self._assert_initialized()
        self._c.run(args=utils.cast_args(args))

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
from .utils import defaults_to


class Kernel:
    def __init__(self, handle=None):
        self._handle = handle

    @defaults_to(False)
    def is_initialized(self):
        return True

    @defaults_to(None)
    def free(self):
        c.kernel.free(self._handle)

    @property
    @defaults_to(None)
    def device(self):
        return c.kernel.device(self._handle)

    @property
    @defaults_to('')
    def mode(self):
        return c.kernel.mode(self._handle)

    @property
    @defaults_to({})
    def properties(self):
        return json.loads(c.kernel.properties(self._handle))

    @property
    @defaults_to('')
    def name(self):
        return c.kernel.name(self._handle)

    @property
    @defaults_to('')
    def source_filename(self):
        return c.kernel.source_filename(self._handle)

    @property
    @defaults_to('')
    def binary_filename(self):
        return c.kernel.binary_filename(self._handle)

    @property
    @defaults_to(0)
    def max_dims(self):
        return c.kernel.max_dims(self._handle)

    @property
    @defaults_to([0, 0, 0])
    def max_outer_dims(self):
        return c.kernel.max_outer_dims(self._handle)

    @property
    @defaults_to([0, 0, 0])
    def max_inner_dims(self):
        return c.kernel.max_inner_dims(self._handle)

    @defaults_to(None)
    def set_run_dims(self, outer_dims, inner_dims):
        c.kernel.set_run_dims(self._handle,
                              outer_dims,
                              inner_dims)

    @defaults_to(None)
    def __call__(self):
        pass

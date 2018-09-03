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


class Memory:
    def __init__(self, handle=None):
        self._handle = handle

    @defaults_to(None)
    def free(self):
        c.memory.free(self._handle)

    @property
    @defaults_to(None)
    def device(self):
        return c.memory.device(self._handle)

    @property
    @defaults_to('')
    def mode(self):
        return c.memory.mode(self._handle)

    @property
    @defaults_to(0)
    def size(self):
        return c.memory.size(self._handle)

    @property
    @defaults_to({})
    def properties(self):
        return json.loads(c.memory.properties(self._handle))

    @defaults_to(None)
    def copy_to(self, dest, bytes, offset, props):
        c.memory.copy_to(self._handle,
                         dest._handle,
                         bytes,
                         offset,
                         json.dumps(props))

    @defaults_to(None)
    def copy_from(self, src, bytes, offset, props):
        c.memory.copy_from(self._handle,
                         src._handle,
                         bytes,
                         offset,
                         json.dumps(props))

    @defaults_to(None)
    def clone(self):
        return c.memory.clone(self._handle)

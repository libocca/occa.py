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


class Memory:
    def __init__(self, c_memory=None):
        if c_memory:
            utils.assert_c_memory(c_memory)
            self._c = c_memory
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.Memory is not initialized')

    def _to_occa_kernel_arg(self):
        return self._c

    @property
    def is_initialized(self):
        '''Return if the memory has been initialized'''
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
    def size(self):
        self._assert_initialized()
        return self._c.size()

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(self._c.properties())

    def __getitem__(self, key):
        self._assert_initialized()
        if (not isinstance(key, slice) or
            key.step != 1):
            raise KeyError('Only accepts slices with step of 1(e.g. mem[:-10])')
        return Memory(
            self._c.slice(offset=key.start,
                          bytes=(key.end - key.start))
        )

    def copy_to(self, dest,
                bytes=None,
                src_offset=None,
                dest_offset=None,
                props=None):
        from .base import memcpy

        self._assert_initialized()
        utils.assert_memory_like(dest)
        if bytes is not None:
            utils.assert_int(bytes)
        if dest_offset is not None:
            utils.assert_int(dest_offset)
        if src_offset is not None:
            utils.assert_int(src_offset)
        props = utils.properties(props)

        memcpy(dest=dest,
               src=self,
               bytes=bytes,
               src_offset=src_offset,
               dest_offset=dest_offset,
               props=props)

    def copy_from(self, src,
                  bytes=None,
                  src_offset=None,
                  dest_offset=None,
                  props=None):
        from .base import memcpy

        self._assert_initialized()
        utils.assert_memory_like(src)
        if bytes is not None:
            utils.assert_int(bytes)
        if dest_offset is not None:
            utils.assert_int(dest_offset)
        if src_offset is not None:
            utils.assert_int(src_offset)
        props = utils.properties(props)

        memcpy(dest=self,
               src=src,
               bytes=bytes,
               src_offset=src_offset,
               dest_offset=dest_offset,
               props=props)

    def clone(self):
        self._assert_initialized()
        return Memory(self._c.clone())

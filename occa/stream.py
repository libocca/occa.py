import json

from . import utils
from .exceptions import UninitializedError


class Stream:
    def __init__(self, c_stream=None):
        if c_stream:
            utils.assert_c_stream(c_stream)
            self._c = c_stream
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.Stream is not initialized')

    @property
    def is_initialized(self):
        '''Return if the stream has been initialized'''
        return (self._c is not None and
                self._c.is_initialized())

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

    def __bool__(self):
        return self.is_initialized

    def __eq__(self, other):
        self._assert_initialized()
        if not isinstance(other, Stream):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        self._assert_initialized()
        return self._c.ptr_as_long()

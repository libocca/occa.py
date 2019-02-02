from . import utils
from .exceptions import UninitializedError


class StreamTag:
    def __init__(self, c_streamtag=None):
        if c_streamtag:
            utils.assert_c_streamtag(c_streamtag)
            self._c = c_streamtag
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.StreamTag is not initialized')

    @property
    def is_initialized(self):
        '''Return if the stream tag has been initialized'''
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

    def wait(self):
        self._assert_initialized()
        self._c.wait()

    def __bool__(self):
        return self.is_initialized

    def __eq__(self, other):
        self._assert_initialized()
        if not isinstance(other, StreamTag):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        self._assert_initialized()
        return self._c.ptr_as_long()

import json
import numpy as np

from . import utils
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

    def __bool__(self):
        return self.is_initialized

    def __call__(self, *args):
        self._assert_initialized()

        args = list(args)
        np_array_args = {
            index: arg
            for index, arg in enumerate(args)
            if isinstance(arg, np.ndarray)
        }
        if len(np_array_args) == 0:
            self._c.run(args=utils.cast_args(args))
            return

        # Copy values to device
        device = self.device
        for index, array in np_array_args.items():
            args[index] = device.malloc(array)

        self._c.run(args=utils.cast_args(args))

        # Copy device values back to host
        for index, array in np_array_args.items():
            args[index].copy_to(array)
            args[index].free()

    def __eq__(self, other):
        self._assert_initialized()
        if not isinstance(other, Kernel):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        self._assert_initialized()
        return self._c.ptr_as_long()

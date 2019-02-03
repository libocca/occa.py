from ..base import get_device
from .py2okl import py2okl


class Kernel:
    def __init__(self, func):
        self.func = func
        self._okl_source = None
        self._kernels = dict()

    @property
    def __name__(self):
        return self.func.__name__

    @property
    def __okl_source__(self):
        if self._okl_source is None:
            self._okl_source = py2okl(self.func)
        return self._okl_source

    def build(self, device, props=None):
        kernel = self._kernels.get(device)
        if kernel is None or not kernel.is_initialized:
            kernel = (
                device.build_kernel_from_string(self.__okl_source__,
                                                self.__name__,
                                                props)
            )
            self._kernels[device] = kernel
        return kernel

    def __call__(self, *args, props=None):
        return self.build(get_device(), props=props)(*args)

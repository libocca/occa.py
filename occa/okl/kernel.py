from ..base import get_device
from .utils import py2okl


class Kernel:
    def __init__(self, func):
        self.func = func
        self._source = None
        self._hashed_sources = dict()
        self._kernels = dict()

    @property
    def __name__(self):
        return self.func.__name__

    def __str__(self):
        return "<occa.okl.kernel.Kernel '{name}'>".format(name=self.__name__)

    def __repr__(self):
        return self.__str__()

    def hash_value(self, value):
        if isinstance(value, dict):
            return hash(frozenset(value.items()))
        return hash(value)

    def hash_values(self, *values):
        hash((self.hash_value(val)
              for val in values
              if val is not None))

    def source(self, *, globals=None, _show_full_stacktrace=False):
        if not globals:
            if not self._source:
                self._source = py2okl(self.func, _show_full_stacktrace=_show_full_stacktrace)
            return self._source

        globals_hash = self.hash_value(globals)
        source = self._hashed_sources.get(globals_hash)
        if source is None:
            source = py2okl(self.func,
                            globals=globals,
                            _show_full_stacktrace=_show_full_stacktrace)
            self._hashed_sources[globals_hash] = source
        return source

    def build(self, device, props=None, globals=None):
        kernel_hash = self.hash_values(device, props, globals)
        kernel = self._kernels.get(kernel_hash)
        if kernel is None or not kernel.is_initialized:
            kernel = device.build_kernel_from_string(self.source(globals=globals),
                                                     self.__name__,
                                                     props)
            self._kernels[kernel_hash] = kernel
        return kernel

    def __call__(self, *args, device=None, props=None, globals=None):
        return self.build(device or get_device(),
                          props=props,
                          globals=globals)(*args)

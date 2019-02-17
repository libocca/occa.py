import json
import numpy as np

from . import c, utils
from . import dtype as occa_dtype
from .exceptions import UninitializedError


class Device:
    def __init__(self, props=None, **kwargs):
        if isinstance(props, c.Device):
            self._c = props
            return

        utils.assert_properties(props, **kwargs)
        props = utils.properties(props, **kwargs)

        if props is not None:
            self._c = c.Device(props=props)
        else:
            self._c = None

    def _assert_initialized(self):
        if not self.is_initialized:
            raise UninitializedError('occa.Device is not initialized')

    @property
    def is_initialized(self):
        '''Return if the device has been initialized'''
        return (self._c is not None and
                self._c.is_initialized())

    def free(self):
        self._assert_initialized()
        self._c.free()

    @property
    def mode(self):
        self._assert_initialized()
        return self._c.mode()

    @property
    def properties(self):
        self._assert_initialized()
        return json.loads(self._c.properties())

    @property
    def kernel_properties(self):
        self._assert_initialized()
        return json.loads(self._c.kernel_properties())

    @property
    def memory_properties(self):
        self._assert_initialized()
        return json.loads(self._c.memory_properties())

    @property
    def memory_size(self):
        self._assert_initialized()
        return self._c.memory_size()

    @property
    def memory_allocated(self):
        self._assert_initialized()
        return self._c.memory_allocated()

    def finish(self):
        self._assert_initialized()
        self._c.finish()

    @property
    def has_separate_memory_space(self):
        self._assert_initialized()
        return self._c.has_separate_memory_space()

    #---[ Stream ]----------------------
    def create_stream(self):
        from .stream import Stream

        self._assert_initialized()
        return Stream(self._c.create_stream())

    @property
    def stream(self):
        from .stream import Stream

        self._assert_initialized()
        return Stream(self._c.get_stream())

    def set_stream(self, stream):
        self._assert_initialized()
        utils.assert_stream(stream)

        self._c.set_stream(stream._c)

    def tag_stream(self):
        from .streamtag import StreamTag

        self._assert_initialized()

        return StreamTag(self._c.tag_stream())

    def wait_for(self, tag):
        self._assert_initialized()
        utils.assert_streamtag(tag)

        self._c.wait_for(tag._c)

    def time_between(self, start, end):
        self._assert_initialized()
        utils.assert_streamtag(start)
        utils.assert_streamtag(end)

        return self._c.time_between(start._c, end._c)
    #===================================

    #---[ Kernel ]----------------------
    def build_kernel(self, filename, kernel, props=None):
        from .kernel import Kernel

        self._assert_initialized()
        utils.assert_str(filename)
        utils.assert_str(kernel)
        props = utils.properties(props) or ''

        return Kernel(
            self._c.build_kernel(filename=filename,
                                 kernel=kernel,
                                 props=props)
        )

    def build_kernel_from_string(self, source, kernel, props=None):
        from .kernel import Kernel

        self._assert_initialized()
        utils.assert_str(source)
        utils.assert_str(kernel)
        props = utils.properties(props) or ''

        return Kernel(
            self._c.build_kernel_from_string(source=source,
                                             kernel=kernel,
                                             props=props)
        )
    #===================================

    #---[ Memory ]----------------------
    def malloc(self, src_or_entries, dtype=None, props=None):
        from .memory import Memory

        self._assert_initialized()
        utils.assert_malloc_src(src_or_entries)
        props = utils.properties(props)

        src = None

        # Convert iterable into np.array
        is_ndarray = isinstance(src_or_entries, np.ndarray)
        if (not is_ndarray and
            isinstance(src_or_entries, list)):
            src_or_entries = np.array(src_or_entries)
            is_ndarray = True

        if is_ndarray:
            src = src_or_entries
            dtype = dtype or src.dtype

            entries = src.size
            if dtype != src.dtype:
                src = src.astype(dtype)
        else:
            entries = src_or_entries

        if dtype:
            dtype = np.dtype(dtype)
        utils.assert_valid_dtype(dtype)

        return Memory(
            self._c.malloc(entries=entries,
                           src=src,
                           dtype=occa_dtype.get_occa_dtype(dtype),
                           props=props),
            dtype=dtype,
        )
    #===================================

    def __bool__(self):
        return self.is_initialized

    def __eq__(self, other):
        self._assert_initialized()
        if not isinstance(other, Device):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        self._assert_initialized()
        return self._c.ptr_as_long()

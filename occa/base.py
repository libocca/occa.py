import json
import numpy as np

from . import c, utils, exceptions


def settings():
    return json.loads(c.settings())


def set_setting(key, value):
    utils.assert_str(key)
    c.set_setting(key=key, value=json.dumps(value))


#---[ Device ]--------------------------
def host():
    from .device import Device

    return Device(c.host())


def get_device():
    from .device import Device

    return Device(c.get_device())


def set_device(device_or_props=None, **kwargs):
    from .device import Device

    if isinstance(device_or_props, Device):
        c.set_device(device=device_or_props._c)
        return

    utils.assert_properties(device_or_props, **kwargs)
    props = utils.properties(device_or_props, **kwargs)

    if props:
        c.set_device(props=props)


def finish():
    get_device().finish()


def create_stream():
    return get_device().create_stream()


def get_stream():
    return get_device().stream


def set_stream(stream):
    get_device().set_stream(stream)


def tag_stream():
    return get_device().tag_stream()


def wait_for(tag):
    return get_device().wait_for(tag)


def time_between(start, end):
    return get_device().time_between(start, end)
#=======================================


#---[ Kernel ]--------------------------
def build_kernel(filename, kernel, props=None):
    return get_device().build_kernel(filename=filename,
                                     kernel=kernel,
                                     props=props)


def build_kernel_from_string(source, kernel, props=None):
    return get_device().build_kernel_from_string(source=source,
                                                 kernel=kernel,
                                                 props=props)
#=======================================


#---[ Memory ]--------------------------
def malloc(src_or_entries, dtype=None, props=None):
    return get_device().malloc(src_or_entries,
                               dtype=dtype,
                               props=props)


def memcpy(dest, src,
           entries=None,
           dest_offset=0,
           src_offset=0,
           dest_buffer=None,
           src_buffer=None,
           props=None):
    from .memory import Memory

    # Verify inputs
    utils.assert_memory_like(dest)
    utils.assert_memory_like(src)

    if isinstance(dest, list):
        dest = np.array(dest)
    if isinstance(src, list):
        src = np.array(src)

    if (not isinstance(dest, Memory) and
        not isinstance(src, Memory)):
        np.copyto(dest, src, casting='unsafe')
        return

    # Verify offsets and props
    utils.assert_int(dest_offset)
    utils.assert_int(src_offset)
    props = utils.properties(props)

    # Find entries
    if (entries is None and
        (len(src) - src_offset) != (len(dest) - dest_offset)):
        raise ValueError('Entries is ambiguous since dest and src lengths differ')

    entries = entries or len(src)

    # Get buffers
    if src.dtype != dest.dtype:
        src_buffer = utils.memory_buffer(src,
                                         entries=entries,
                                         buffer=src_buffer)
        dest_buffer = utils.memory_buffer(dest,
                                          entries=entries,
                                          buffer=dest_buffer)
    else:
        src_buffer = dest_buffer = None

    src_bytes = entries * src.dtype.itemsize
    dest_bytes = entries * dest.dtype.itemsize

    # Extract out the C types
    if isinstance(dest, Memory):
        dest = dest._c
    if isinstance(src, Memory):
        src = src._c

    # M: occa.Memory
    # A: numpy.ndarray
    # Types 1 or 2
    #
    # | src | src_buffer | dest_buffer | dest |
    # +-----+------------+-------------+------+
    # | M1  | None       | None        | M1   |
    # | M1  | A1         | None        | A2   |
    # | M1  | A1         | A2          | M2   |
    # | A1  | None       | A2          | M2   |
    if src_buffer is not None:
        c.memcpy(dest=src_buffer,
                 src=src,
                 bytes=src_bytes,
                 dest_offset=0,
                 src_offset=src_offset,
                 props=props)
        src = src_buffer
        src_offset = 0

    if dest_buffer is not None:
        np.copyto(dest_buffer, src, casting='unsafe')
        src = dest_buffer

    if (isinstance(dest, np.ndarray) and
        isinstance(src, np.ndarray)):
        np.copyto(dest, src, casting='unsafe')
        return

    c.memcpy(dest=dest,
             src=src,
             bytes=dest_bytes,
             dest_offset=dest_offset,
             src_offset=src_offset,
             props=props)
#=======================================

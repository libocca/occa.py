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

def settings():
    return json.loads(c.settings())

def print_mode_info():
    c.print_mode_info()

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
        c.set_device(device_or_props._c)
        return

    utils.assert_properties(device_or_props, **kwargs)
    props = utils.properties(device_or_props, **kwargs)

    if props:
        c.set_device(props=props)
    else:
        return None

def finish():
    c.finish()

def create_stream():
    from .stream import Stream

    return Stream(c.create_stream())

def get_stream():
    from .stream import Stream

    return Stream(c.get_stream())

def set_stream(stream):
    from .stream import Stream

    utils.assert_stream(stream)
    c.set_stream(_c)

def tag_stream():
    from .streamtag import StreamTag

    return StreamTag(c.tag_stream())

def wait_for_tag(tag):
    utils.assert_streamtag(tag)
    self._c.wait_for(tag._c)

def time_between_tags(start, end):
    utils.assert_streamtag(start)
    utils.assert_streamtag(end)

    return c.time_between(start._c, end._c)
#=======================================

#---[ Kernel ]--------------------------
def build_kernel(filename, kernel, props=None):
    from .kernel import Kernel

    utils.assert_str(filename)
    utils.assert_str(kernel)
    props = utils.properties(props) or ''

    return Kernel(
        c.build_kernel(filename=filename,
                       kernel=kernel,
                       props=props)
    )

def build_kernel_from_string(source, kernel, props=None):
    from .kernel import Kernel

    utils.assert_str(source)
    utils.assert_str(kernel)
    props = utils.properties(props) or ''

    return Kernel(
        c.build_kernel_from_string(source=source,
                                   kernel=kernel,
                                   props=props)
    )
#=======================================

#---[ Memory ]--------------------------
def malloc(self, bytes, src=None, props=None):
    from .memory import Memory

    utils.assert_int(bytes)
    props = utils.properties(props) or ''

    raise NotImplementedError

    return Memory(
        c.malloc(bytes=bytes,
                 src=src,
                 props=props)
    )

def memcpy(dest, src,
           bytes=None,
           dest_offset=None, src_offset=None,
           props=None):
    from .memory import Memory

    utils.assert_memory_like(dest)
    utils.assert_memory_like(src)
    if bytes is not None:
        utils.assert_int(bytes)
    if dest_offset is not None:
        utils.assert_int(dest_offset)
    if src_offset is not None:
        utils.assert_int(src_offset)
    props = utils.properties(props)

    if isinstance(dest, Memory):
        dest = dest._c
    if isinstance(src, Memory):
        src = src._c

    c.memcpy(dest=dest,
             src=src,
             dest_offset=dest_offset,
             src_offset=src_offset,
             props=props)
#=======================================

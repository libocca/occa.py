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

from . import c, utils, device, memory, kernel as K, stream, streamtag

def settings():
    return json.loads(c.settings())

def print_mode_info():
    c.print_mode_info()

#---[ Device ]--------------------------
def host():
    return device.Device(c.host())

def get_device():
    return device.Device(c.get_device())

def set_device(device_or_props=None, **kwargs):
    if isinstance(device_or_props, device.Device):
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
    return stream.Stream(c.create_stream())

def get_stream():
    return stream.Stream(c.get_stream())

def set_stream(stream):
    utils.assert_stream(stream)
    c.set_stream(stream._c)

def tag_stream():
    return streamtag.StreamTag(c.tag_stream())

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
    utils.assert_str(filename)
    utils.assert_str(kernel)
    props = utils.properties(props) or ''

    return K.Kernel(
        c.build_kernel(filename=filename,
                       kernel=kernel,
                       props=props)
    )

def build_kernel_from_string(source, kernel, props=None):
    utils.assert_str(source)
    utils.assert_str(kernel)
    props = utils.properties(props) or ''

    return K.Kernel(
        c.build_kernel_from_string(source=source,
                                   kernel=kernel,
                                   props=props)
    )
#=======================================

#---[ Memory ]--------------------------
def malloc(self, bytes, src=None, props=None):
    utils.assert_int(bytes)
    props = utils.properties(props) or ''

    raise NotImplementedError

    return memory.Memory(
        c.malloc(bytes=bytes,
                 src=src,
                 props=props)
    )

def memcpy(dest, src, *,
           bytes, dest_offset, src_offset, props):
    raise NotImplementedError
#=======================================

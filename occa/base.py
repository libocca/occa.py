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
from .device import Device
from .kernel import Kernel
from .memory import Memory

#---[ Device ]--------------------------
def host():
    pass

def get_device():
    pass

def set_device(device=None, **kwargs):
    if device is None:
        device = kwargs
    if isinstance(device, str):
        device = json.loads(device)

    if isinstance(device, dict):
        pass
    elif isinstance(device, device.Device):
        pass
    else:
        raise ValueError('Expected a str, dict, occa.Device, or **kwargs')

def device_properties():
    pass

def finish():
    pass

def wait_for_tag(tag):
    pass

def time_between_tags(start_tag, end_tag):
    pass

def create_stream():
    pass

def get_stream():
    pass

def set_stream():
    pass

def tag_stream():
    pass
#=======================================

#---[ Kernel ]--------------------------
def build_kernel(filename, kernel_name, props):
    pass

def build_kernel_from_string(source, kernel_name, props):
    pass
#=======================================

#---[ Memory ]--------------------------
def malloc(bytes, src, props):
    pass

def umalloc(bytes, src, props):
    pass

def memcpy(dest, src, *,
           bytes, dest_offset, src_offset, props):
    pass
#=======================================

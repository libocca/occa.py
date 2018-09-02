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
class Device:
    def __init__(self, props=None, **kwargs):
        if props is None:
            props = kwargs
        if isinstance(props, str):
            props = json.loads(props)

        if isinstance(props, dict):
            pass
        else:
            raise ValueError('Expected a str, dict, or **kwargs')

    def is_initialized(self):
        pass

    def free(self):
        pass

    @property
    def mode(self):
        pass

    @property
    def properties(self):
        pass

    @property
    def kernel_properties(self):
        pass

    @property
    def memory_properties(self):
        pass

    def memory_size(self):
        pass

    def memory_allocated(self):
        pass

    def finish(self):
        pass

    @property
    def has_separate_memory_space(self):
        pass

    #---[ Stream ]----------------------
    def create_stream(self):
        pass

    @property
    def stream(self):
        pass

    def set_stream(self, stream):
        pass

    def free_stream(self, stream):
        pass

    def tag_stream(self):
        pass

    def wait_for_tag(self, tag):
        pass

    def time_between_tags(self, start_tag, end_tag):
        pass
    #===================================

    #---[ Kernel ]----------------------
    def build_kernel(self, filename, kernel_name, props):
        pass

    def build_kernel_from_string(self, source, kernel_name, props):
        pass
    #===================================

    #---[ Memory ]----------------------
    def malloc(self, bytes, src, props):
        pass

    def umalloc(self, bytes, src, props):
        pass
    #===================================

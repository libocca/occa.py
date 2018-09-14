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
import contextlib
import io
import occa
import os
import pytest
import time

import numpy as np


def add_vectors_file():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     'add_vectors.okl')
    )


def test_settings():
    settings = occa.settings()
    assert isinstance(settings, dict)
    assert 'version' in settings
    assert 'okl-version' in settings


def test_print_mode_info():
    occa.print_mode_info()


#---[ Device ]--------------------------
def test_host():
    host = occa.host()

    assert isinstance(host, occa.Device)
    assert host.is_initialized == True
    assert host.mode == 'Serial'


def test_get_device():
    device = occa.get_device()

    assert isinstance(device, occa.Device)
    assert device.is_initialized == True
    assert device.mode == 'Serial'


def test_set_device():
    host = occa.host()

    assert host == occa.get_device()

    occa.set_device(mode='Serial')
    dev = occa.get_device()

    assert isinstance(dev, occa.Device)
    assert host != dev

    occa.set_device(host)
    dev = occa.get_device()

    assert isinstance(dev, occa.Device)
    assert host == dev


def test_finish():
    occa.finish()


def test_create_stream():
    stream1 = occa.get_stream()
    stream2 = occa.create_stream()

    assert isinstance(stream2, occa.Stream)
    assert stream2.is_initialized
    assert stream1 != stream2


def test_get_stream():
    stream = occa.get_stream()

    assert isinstance(stream, occa.Stream)


def test_set_stream():
    host = occa.host()
    stream = host.stream

    assert host == occa.get_device()

    stream2 = host.create_stream()

    assert stream != stream2

    occa.set_stream(stream2)

    assert stream != occa.get_stream()


def test_tags():
    # TODO: Fix
    return
    start = occa.tag_stream()
    time.sleep(0.5)
    end = occa.tag_stream()

    occa.wait_for_tag(end)

    assert isinstance(start, occa.StreamTag)
    assert occa.time_between(start, end) > 0.4
#=======================================


#---[ Kernel ]--------------------------
def test_build_kernel():
    k = occa.build_kernel(
        add_vectors_file(),
        'add_vectors',
    )
    assert isinstance(k, occa.Kernel)

    k = occa.build_kernel(
        add_vectors_file(),
        'add_vectors',
        {
            'define': {
                'FOO': 1,
                'BAR': '1',
            },
        },
    )
    assert isinstance(k, occa.Kernel)

    k = occa.build_kernel(
        add_vectors_file(),
        'add_vectors',
        'define: { FOO: 1, BAR: "" }',
    )
    assert isinstance(k, occa.Kernel)


def test_build_kernel_from_string():
    with open(add_vectors_file()) as f:
        source = f.read()

    k = occa.build_kernel_from_string(
        source,
        'add_vectors',
    )
    assert isinstance(k, occa.Kernel)

    k = occa.build_kernel_from_string(
        source,
        'add_vectors',
        {
            'define': {
                'FOO': 1,
                'BAR': '1',
            },
        },
    )
    assert isinstance(k, occa.Kernel)

    k = occa.build_kernel_from_string(
        source,
        'add_vectors',
        'define: { FOO: 1, BAR: "" }',
    )
    assert isinstance(k, occa.Kernel)
#=======================================


#---[ Memory ]--------------------------
def test_malloc():
    m = occa.malloc(10, dtype=np.int32)

    assert isinstance(m, occa.Memory)
    assert m.dtype == np.int32
    assert len(m) == 10

    m = occa.malloc(np.zeros(5, dtype=np.float32))

    assert isinstance(m, occa.Memory)
    assert m.dtype == np.float32
    assert len(m) == 5


def test_memcpy():
    def zeros(dtype):
        return np.zeros(5, dtype=dtype)
    def ones(dtype):
        return np.ones(5, dtype=dtype)
    def m_zeros(dtype):
        return occa.malloc(zeros(dtype))
    def m_ones(dtype):
        return occa.malloc(ones(dtype))

    # D# -> D#
    d1 = ones(np.float32)
    d2 = zeros(np.int32)
    occa.memcpy(d2, d1)

    assert d2.dtype == np.int32
    assert all(d2)

    # M1 -> D1
    m = m_ones(np.float32)
    d = zeros(np.int32)
    occa.memcpy(d, m)

    assert d.dtype == np.int32
    assert all(d)

    # M1 -> D2
    m = m_ones(np.float32)
    d = zeros(np.int32)
    occa.memcpy(d, m)

    assert d.dtype == np.int32
    assert all(d)

    # D1 -> M1
    d = ones(np.float32)
    m = m_zeros(np.int32)
    occa.memcpy(m, d)

    assert m.dtype == np.int32
    assert all(m.to_ndarray())

    # D1 -> M2
    d = ones(np.float32)
    m = m_zeros(np.int32)
    occa.memcpy(m, d)

    assert m.dtype == np.int32
    assert all(m.to_ndarray())

    # M1 -> M1
    m1 = m_ones(np.float32)
    m2 = m_zeros(np.int32)
    occa.memcpy(m2, m1)

    assert m2.dtype == np.int32
    assert all(m.to_ndarray())

    # M1 -> M2
    m1 = m_ones(np.float32)
    m2 = m_zeros(np.int32)
    occa.memcpy(m2, m1)

    assert m2.dtype == np.int32
    assert all(m.to_ndarray())
#=======================================

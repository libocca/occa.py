/* The MIT License (MIT)
 *
 * Copyright (c) 2018 David Medina
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include "header.h"


//---[ Globals & Flags ]----------------
static PyObject* py_occa_settings(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_print_mode_info(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Device ]-------------------------
static PyObject* py_occa_host(PyObject *self,
                              PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_get_device(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_set_device(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_set_device_from_string(PyObject *self,
                                                PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_device_properties(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_load_kernels(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_finish(PyObject *self,
                                PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_create_stream(PyObject *self,
                                       PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_get_stream(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_set_stream(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_tag_stream(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_wait_for_tag(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_time_between_tags(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Kernel ]-------------------------
static PyObject* py_occa_build_kernel(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_build_kernel_from_string(PyObject *self,
                                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_build_kernel_from_binary(PyObject *self,
                                                  PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Memory ]-------------------------
static PyObject* py_occa_malloc(PyObject *self,
                                PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_umalloc(PyObject *self,
                                 PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


OCCA_PY_MODULE(
  base,
  OCCA_PY_METHOD(settings),
  OCCA_PY_METHOD(print_mode_info),
  OCCA_PY_METHOD(host),
  OCCA_PY_METHOD(get_device),
  OCCA_PY_METHOD(set_device),
  OCCA_PY_METHOD(set_device_from_string),
  OCCA_PY_METHOD(device_properties),
  OCCA_PY_METHOD(load_kernels),
  OCCA_PY_METHOD(finish),
  OCCA_PY_METHOD(create_stream),
  OCCA_PY_METHOD(get_stream),
  OCCA_PY_METHOD(set_stream),
  OCCA_PY_METHOD(tag_stream),
  OCCA_PY_METHOD(wait_for_tag),
  OCCA_PY_METHOD(time_between_tags),
  OCCA_PY_METHOD(build_kernel),
  OCCA_PY_METHOD(build_kernel_from_string),
  OCCA_PY_METHOD(build_kernel_from_binary),
  OCCA_PY_METHOD(malloc),
  OCCA_PY_METHOD(umalloc)
);

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


static PyObject* py_occa_create(PyObject *self,
                                PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_is_initialized(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_mode(PyObject *self,
                              PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_properties(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_kernel_properties(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_memory_properties(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_memory_size(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_memory_allocated(PyObject *self,
                                          PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_finish(PyObject *self,
                                PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_has_separate_memory_space(PyObject *self,
                                                   PyObject *args) {
  Py_RETURN_NONE;
}


//---[ Stream ]-------------------------
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
  device,
  OCCA_PY_METHOD(create),
  OCCA_PY_METHOD(is_initialized),
  OCCA_PY_METHOD(mode),
  OCCA_PY_METHOD(properties),
  OCCA_PY_METHOD(kernel_properties),
  OCCA_PY_METHOD(memory_properties),
  OCCA_PY_METHOD(memory_size),
  OCCA_PY_METHOD(memory_allocated),
  OCCA_PY_METHOD(finish),
  OCCA_PY_METHOD(has_separate_memory_space),
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

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
#include "header.hpp"


static PyObject* py_occa_is_initialized(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_properties(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_device(PyObject *self,
                                PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_name(PyObject *self,
                              PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_source_filename(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_binary_filename(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_max_dims(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_max_outer_dims(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_max_inner_dims(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_set_run_dims(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_run_n(PyObject *self,
                               PyObject *args) {
  Py_RETURN_NONE;
}


OCCA_PY_MODULE(
  kernel,
  OCCA_PY_METHOD(is_initialized),
  OCCA_PY_METHOD(properties),
  OCCA_PY_METHOD(device),
  OCCA_PY_METHOD(name),
  OCCA_PY_METHOD(source_filename),
  OCCA_PY_METHOD(binary_filename),
  OCCA_PY_METHOD(max_dims),
  OCCA_PY_METHOD(max_outer_dims),
  OCCA_PY_METHOD(max_inner_dims),
  OCCA_PY_METHOD(set_run_dims),
  OCCA_PY_METHOD(run_n)
);

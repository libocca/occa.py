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

static PyObject* py_occa_is_managed(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_start_managing(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_stop_managing(PyObject *self,
                                       PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_sync_to_device(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_sync_to_host(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_needs_sync(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_sync(PyObject *self,
                              PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_dont_sync(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occa_free_uva_ptr(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}


OCCA_PY_MODULE(
  uva,
  OCCA_PY_METHOD(is_managed),
  OCCA_PY_METHOD(start_managing),
  OCCA_PY_METHOD(stop_managing),
  OCCA_PY_METHOD(sync_to_device),
  OCCA_PY_METHOD(sync_to_host),
  OCCA_PY_METHOD(needs_sync),
  OCCA_PY_METHOD(sync),
  OCCA_PY_METHOD(dont_sync),
  OCCA_PY_METHOD(free_uva_ptr)
);

/* The MIT License (MIT)
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


static PyObject* py_occaKernelIsInitialized(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelGetProperties(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelGetDevice(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelName(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelSourceFilename(PyObject *self,
                                             PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelBinaryFilename(PyObject *self,
                                             PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelMaxDims(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelMaxOuterDims(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelMaxInnerDims(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelSetRunDims(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaKernelRunN(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}


OCCA_PY_MODULE(
  kernel,
  OCCA_PY_METHOD(occaKernelIsInitialized),
  OCCA_PY_METHOD(occaKernelGetProperties),
  OCCA_PY_METHOD(occaKernelGetDevice),
  OCCA_PY_METHOD(occaKernelName),
  OCCA_PY_METHOD(occaKernelSourceFilename),
  OCCA_PY_METHOD(occaKernelBinaryFilename),
  OCCA_PY_METHOD(occaKernelMaxDims),
  OCCA_PY_METHOD(occaKernelMaxOuterDims),
  OCCA_PY_METHOD(occaKernelMaxInnerDims),
  OCCA_PY_METHOD(occaKernelSetRunDims),
  OCCA_PY_METHOD(occaKernelRunN)
);

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
static PyObject* py_occaSettings(PyObject *self,
                                 PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaPrintModeInfo(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Device ]-------------------------
static PyObject* py_occaHost(PyObject *self,
                             PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaGetDevice(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaSetDevice(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaSetDeviceFromString(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceProperties(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaLoadKernels(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaFinish(PyObject *self,
                               PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaCreateStream(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaGetStream(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaSetStream(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaTagStream(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaWaitForTag(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaTimeBetweenTags(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Kernel ]-------------------------
static PyObject* py_occaBuildKernel(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaBuildKernelFromString(PyObject *self,
                                              PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaBuildKernelFromBinary(PyObject *self,
                                              PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Memory ]-------------------------
static PyObject* py_occaMalloc(PyObject *self,
                               PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaUMalloc(PyObject *self,
                                PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


OCCA_PY_MODULE(
  base,
  OCCA_PY_METHOD(occaSettings),
  OCCA_PY_METHOD(occaPrintModeInfo),
  OCCA_PY_METHOD(occaHost),
  OCCA_PY_METHOD(occaGetDevice),
  OCCA_PY_METHOD(occaSetDevice),
  OCCA_PY_METHOD(occaSetDeviceFromString),
  OCCA_PY_METHOD(occaDeviceProperties),
  OCCA_PY_METHOD(occaLoadKernels),
  OCCA_PY_METHOD(occaFinish),
  OCCA_PY_METHOD(occaCreateStream),
  OCCA_PY_METHOD(occaGetStream),
  OCCA_PY_METHOD(occaSetStream),
  OCCA_PY_METHOD(occaTagStream),
  OCCA_PY_METHOD(occaWaitForTag),
  OCCA_PY_METHOD(occaTimeBetweenTags),
  OCCA_PY_METHOD(occaBuildKernel),
  OCCA_PY_METHOD(occaBuildKernelFromString),
  OCCA_PY_METHOD(occaBuildKernelFromBinary),
  OCCA_PY_METHOD(occaMalloc),
  OCCA_PY_METHOD(occaUMalloc)
);

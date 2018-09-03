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


static PyObject* py_occaCreateDevice(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaCreateDeviceFromString(PyObject *self,
                                               PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceIsInitialized(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceMode(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceGetProperties(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceGetKernelProperties(PyObject *self,
                                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceGetMemoryProperties(PyObject *self,
                                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceMemorySize(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceMemoryAllocated(PyObject *self,
                                              PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceFinish(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceHasSeparateMemorySpace(PyObject *self,
                                                     PyObject *args) {
  Py_RETURN_NONE;
}


//---[ Stream ]-------------------------
static PyObject* py_occaDeviceCreateStream(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceGetStream(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceSetStream(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceTagStream(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceWaitForTag(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceTimeBetweenTags(PyObject *self,
                                              PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Kernel ]-------------------------
static PyObject* py_occaDeviceBuildKernel(PyObject *self,
                                          PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceBuildKernelFromString(PyObject *self,
                                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceBuildKernelFromBinary(PyObject *self,
                                                    PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


//---[ Memory ]-------------------------
static PyObject* py_occaDeviceMalloc(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDeviceUMalloc(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}
//======================================


OCCA_PY_MODULE(
  device,
  OCCA_PY_METHOD(occaCreateDevice),
  OCCA_PY_METHOD(occaCreateDeviceFromString),
  OCCA_PY_METHOD(occaDeviceIsInitialized),
  OCCA_PY_METHOD(occaDeviceMode),
  OCCA_PY_METHOD(occaDeviceGetProperties),
  OCCA_PY_METHOD(occaDeviceGetKernelProperties),
  OCCA_PY_METHOD(occaDeviceGetMemoryProperties),
  OCCA_PY_METHOD(occaDeviceMemorySize),
  OCCA_PY_METHOD(occaDeviceMemoryAllocated),
  OCCA_PY_METHOD(occaDeviceFinish),
  OCCA_PY_METHOD(occaDeviceHasSeparateMemorySpace),
  OCCA_PY_METHOD(occaDeviceCreateStream),
  OCCA_PY_METHOD(occaDeviceGetStream),
  OCCA_PY_METHOD(occaDeviceSetStream),
  OCCA_PY_METHOD(occaDeviceTagStream),
  OCCA_PY_METHOD(occaDeviceWaitForTag),
  OCCA_PY_METHOD(occaDeviceTimeBetweenTags),
  OCCA_PY_METHOD(occaDeviceBuildKernel),
  OCCA_PY_METHOD(occaDeviceBuildKernelFromString),
  OCCA_PY_METHOD(occaDeviceBuildKernelFromBinary),
  OCCA_PY_METHOD(occaDeviceMalloc),
  OCCA_PY_METHOD(occaDeviceUMalloc)
);

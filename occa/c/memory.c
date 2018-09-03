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


static PyObject* py_occaMemoryIsInitialized(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryPtr(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryGetDevice(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryGetProperties(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemorySize(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemorySlice(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

//---[ UVA ]----------------------------
static PyObject* py_occaMemoryIsManaged(PyObject *self,
                                        PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryInDevice(PyObject *self,
                                       PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryIsStale(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryStartManaging(PyObject *self,
                                            PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryStopManaging(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemorySyncToDevice(PyObject *self,
                                           PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemorySyncToHost(PyObject *self,
                                         PyObject *args) {
  Py_RETURN_NONE;
}
//======================================

static PyObject* py_occaMemcpy(PyObject *self,
                               PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaCopyMemToMem(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaCopyPtrToMem(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaCopyMemToPtr(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryClone(PyObject *self,
                                    PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaMemoryDetach(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaWrapCpuMemory(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}


OCCA_PY_MODULE(
  memory,
  OCCA_PY_METHOD(occaMemoryIsInitialized),
  OCCA_PY_METHOD(occaMemoryPtr),
  OCCA_PY_METHOD(occaMemoryGetDevice),
  OCCA_PY_METHOD(occaMemoryGetProperties),
  OCCA_PY_METHOD(occaMemorySize),
  OCCA_PY_METHOD(occaMemorySlice),
  OCCA_PY_METHOD(occaMemoryIsManaged),
  OCCA_PY_METHOD(occaMemoryInDevice),
  OCCA_PY_METHOD(occaMemoryIsStale),
  OCCA_PY_METHOD(occaMemoryStartManaging),
  OCCA_PY_METHOD(occaMemoryStopManaging),
  OCCA_PY_METHOD(occaMemorySyncToDevice),
  OCCA_PY_METHOD(occaMemorySyncToHost),
  OCCA_PY_METHOD(occaMemcpy),
  OCCA_PY_METHOD(occaCopyMemToMem),
  OCCA_PY_METHOD(occaCopyPtrToMem),
  OCCA_PY_METHOD(occaCopyMemToPtr),
  OCCA_PY_METHOD(occaMemoryClone),
  OCCA_PY_METHOD(occaMemoryDetach),
  OCCA_PY_METHOD(occaWrapCpuMemory)
);

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


static PyObject* py_occaIsManaged(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaStartManaging(PyObject *self,
                                      PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaStopManaging(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaSyncToDevice(PyObject *self,
                                     PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaSyncToHost(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaNeedsSync(PyObject *self,
                                  PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaSync(PyObject *self,
                             PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaDontSync(PyObject *self,
                                 PyObject *args) {
  Py_RETURN_NONE;
}

static PyObject* py_occaFreeUvaPtr(PyObject *self,
                                   PyObject *args) {
  Py_RETURN_NONE;
}


OCCA_PY_MODULE(
  uva,
  OCCA_PY_METHOD(occaIsManaged),
  OCCA_PY_METHOD(occaStartManaging),
  OCCA_PY_METHOD(occaStopManaging),
  OCCA_PY_METHOD(occaSyncToDevice),
  OCCA_PY_METHOD(occaSyncToHost),
  OCCA_PY_METHOD(occaNeedsSync),
  OCCA_PY_METHOD(occaSync),
  OCCA_PY_METHOD(occaDontSync),
  OCCA_PY_METHOD(occaFreeUvaPtr)
);

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

#ifndef OCCA_PY_HEADER_HEADER
#define OCCA_PY_HEADER_HEADER

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <occa.h>
#include <occa/c/defines.h>

#include <Python.h>
#include "numpy/arrayobject.h"

#define OCCA_PY_METHOD(FUNC)                    \
  {                                             \
    #FUNC,                                      \
    py_##FUNC,                                  \
    METH_VARARGS,                               \
    NULL                                        \
  }


#define OCCA_PY_MODULE(MODULE, ...)                     \
  static PyMethodDef occa_c_##MODULE##_methods[] = {    \
    __VA_ARGS__,                                        \
    {NULL, NULL, 0, NULL}                               \
  };                                                    \
                                                        \
  static PyModuleDef occa_c_##MODULE##_module = {       \
    PyModuleDef_HEAD_INIT,                              \
    "occa.c." #MODULE,                                  \
    "Wrappers for " #MODULE " methods",                 \
    -1,                                                 \
    occa_c_##MODULE##_methods                           \
  };                                                    \
                                                        \
  PyMODINIT_FUNC PyInit_##MODULE() {                    \
    return PyModule_Create(&occa_c_##MODULE##_module);  \
  }

#endif

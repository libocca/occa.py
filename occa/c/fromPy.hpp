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

#ifndef OCCA_PY_FROMPY_HEADER
#define OCCA_PY_FROMPY_HEADER

#include "defines.hpp"
#include "types.hpp"


namespace occa {
  namespace py {
    static bool isString(PyObject *obj) {
#if OCCA_PY3
      return PyUnicode_Check(obj);
#elif OCCA_PY2
      return PyObject_TypeCheck(obj, &PyBaseString_Type);
#endif
    }

    static const char* str(PyObject *obj) {
#if OCCA_PY3
      return PyUnicode_AS_DATA(obj);
#elif OCCA_PY2
      return PyString_AsString(obj);
#endif
    }

    static long long longlong(PyObject *obj) {
      return PyLong_AsLongLong(obj);
    }

    static void* ptr(PyObject *capsule) {
      return PyCapsule_GetPointer(capsule, NULL);
    }
  }
}

#endif

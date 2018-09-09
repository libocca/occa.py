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

#include <occa.hpp>

#include "defines.hpp"
#include "types.hpp"


namespace occa {
  namespace py {
    static bool isNone(void *obj) {
      return (!obj || obj == Py_None);
    }

    static bool isString(PyObject *obj) {
#if OCCA_PY3
      return obj && PyUnicode_Check(obj);
#elif OCCA_PY2
      return obj && PyObject_TypeCheck(obj, &PyBaseString_Type);
#endif
    }

    static bool isMemory(PyObject *obj) {
      return obj && PyObject_TypeCheck(obj, MemoryType());
    }

    static bool isNumpyArray(PyObject *obj) {
      return obj && PyArray_Check(obj);
    }

    static bool isNumpyScalar(PyObject *obj) {
      return obj && PyArray_CheckScalar(obj);
    }

    static std::string str(void *obj) {
      if (!obj) {
        return std::string();
      }
#if OCCA_PY3
      PyObject *pyObj = PyUnicode_AsEncodedString((PyObject*) obj,
                                                  "utf-8",
                                                  "Error ~");
      if (!pyObj) {
        return std::string();
      }
      std::string s = PyBytes_AS_STRING(pyObj);
      Py_XDECREF(pyObj);
      return s;
#elif OCCA_PY2
      return PyString_AsString((PyObject*) obj);
#endif
    }

    static long long longlong(void *obj) {
      return (obj
              ? PyLong_AsLongLong((PyObject*) obj)
              : 0);
    }

    static void* ptr(PyObject *obj) {
      if (isNone(obj)) {
        return NULL;
      }
      if (isNumpyArray(obj)) {
        return PyArray_DATA((PyArrayObject*) obj);
      }
      return PyCapsule_GetPointer(obj, NULL);
    }

    static bool setKernelArg(PyObject *obj,
                             occa::kernelArg &arg) {
      // NULL or None
      if (isNone(obj)) {
        std::cout << "NONE\n";
        arg = occa::kernelArg((void*) NULL);
        return true;
      }

      // occa::memory
      if (isMemory(obj)) {
        std::cout << "MEMORY\n";
        arg = *(((Memory*) obj)->memory);
        return true;
      }

      // numpy dtype
      if (isNumpyScalar(obj)) {
        std::cout << "SCALAR\n";
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        bool setArg = false;

#define CASE_TYPENUM(TYPENUM, SCALARTYPE)       \
        case TYPENUM:                           \
          arg = ((SCALARTYPE*)obj)->obval;      \
          setArg = true;                        \
          break

        switch (descr->type_num) {
          CASE_TYPENUM(NPY_BOOL   , PyBoolScalarObject);
          CASE_TYPENUM(NPY_INT8   , PyInt8ScalarObject);
          CASE_TYPENUM(NPY_UINT8  , PyUInt8ScalarObject);
          CASE_TYPENUM(NPY_INT16  , PyInt16ScalarObject);
          CASE_TYPENUM(NPY_UINT16 , PyUInt16ScalarObject);
          CASE_TYPENUM(NPY_INT32  , PyInt32ScalarObject);
          CASE_TYPENUM(NPY_UINT32 , PyUInt32ScalarObject);
          CASE_TYPENUM(NPY_INT64  , PyInt64ScalarObject);
          CASE_TYPENUM(NPY_UINT64 , PyUInt64ScalarObject);
          CASE_TYPENUM(NPY_FLOAT32, PyFloat32ScalarObject);
          CASE_TYPENUM(NPY_FLOAT64, PyFloat64ScalarObject);
        }
#undef CASE_TYPENUM

        Py_DECREF(descr);

        if (setArg) {
          return true;
        }
      }

      raise("Unsupported type for a kernel argument");
      return false;
    }
  }
}

#endif

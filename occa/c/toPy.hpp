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

#ifndef OCCA_PY_TOPY_HEADER
#define OCCA_PY_TOPY_HEADER

#include "defines.hpp"
#include "types.hpp"


namespace occa {
  namespace py {
    // Special
    static PyObject* None() {
      Py_RETURN_NONE;
    }

    static PyObject* True() {
      Py_RETURN_TRUE;
    }

    static PyObject* False() {
      Py_RETURN_FALSE;
    }

    static PyObject* toPy(void *ptr) {
      return PyCapsule_New(ptr, NULL, NULL);
    }

    // Bool
    PyObject* toPy(const bool b) {
      if (b) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }

    // Integer Types
    PyObject* toPy(const int value) {
      return PyLong_FromLong((long) value);
    }

    PyObject* toPy(const long value) {
      return PyLong_FromLong(value);
    }

    PyObject* toPy(const long long value) {
      return PyLong_FromLong(value);
    }

    PyObject* toPy(const size_t value) {
      return PyLong_FromSize_t(value);
    }

    // Float / Double
    PyObject* toPy(const float value) {
      return PyFloat_FromDouble((double) value);
    }

    PyObject* toPy(const double value) {
      return PyFloat_FromDouble(value);
    }

    // String
    static PyObject* toPy(const char *c) {
      return PyUnicode_FromString(c);
    }

    PyObject* toPy(const std::string s) {
      return PyUnicode_FromString(s.c_str());
    }

    // Core
    static PyObject* newCoreType(PyTypeObject *Type,
                                 void *ptr,
                                 const std::string &name) {

      PyObject *typeObj = occa::py::toPy(ptr);
      PyObject *typeArgs = Py_BuildValue("()");
      PyObject *typeKwargs = Py_BuildValue("{s:O}",
                                           name.c_str(), typeObj);

      PyObject *pyType = PyObject_Call((PyObject*) Type, typeArgs, typeKwargs);

      Py_DECREF(typeObj);
      Py_DECREF(typeArgs);
      Py_DECREF(typeKwargs);

      return pyType;
    }

    PyObject* toPy(const occa::device &device) {
      return newCoreType(occa::py::DeviceType(),
                         (void*) device.getModeDevice(),
                         "device");
    }

    PyObject* toPy(const occa::kernel &kernel) {
      return newCoreType(occa::py::KernelType(),
                         (void*) kernel.getModeKernel(),
                         "kernel");
    }

    PyObject* toPy(const occa::memory &memory) {
      return newCoreType(occa::py::MemoryType(),
                         (void*) memory.getModeMemory(),
                         "memory");
    }

    // Props / JSON
    PyObject* toPy(const occa::properties &props) {
      return toPy(props.dump(0));
    }

    PyObject* toPy(const occa::json &j) {
      return toPy(j.dump(0));
    }
  }
}


#endif

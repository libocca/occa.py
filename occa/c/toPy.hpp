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


namespace occa {
  namespace py {
    static PyTypeObject *Device = NULL;
    static PyTypeObject *Kernel = NULL;
    static PyTypeObject *Memory = NULL;
    static PyTypeObject *Stream = NULL;

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

    static void* ptr(PyObject *capsule) {
      return PyCapsule_GetPointer(capsule, NULL);
    }

    static PyObject* toPy(const void *ptr) {
      return PyCapsule_New(const_cast<void*>(ptr), NULL, NULL);
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
    static PyTypeObject* getTypeFromModule(const std::string &moduleName,
                                           const std::string &className) {
      PyObject *module = PyImport_ImportModule(moduleName.c_str());
      return (PyTypeObject*) PyObject_GetAttrString(module, className.c_str());
    }

    static PyObject* newCoreType(PyTypeObject *Type,
                                 void *ptr) {
      PyObject *args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, toPy(ptr));
      return Type->tp_new(Type, args, NULL);
    }

    PyObject* toPy(const occa::device &device) {
      static PyTypeObject *Device = getTypeFromModule("occa.c.device", "Device");
      return newCoreType(Device, (void*) device.getModeDevice());
    }

    PyObject* toPy(const occa::kernel &kernel) {
      static PyTypeObject *Kernel = getTypeFromModule("occa.c.kernel", "Kernel");
      return newCoreType(Kernel, (void*) kernel.getModeKernel());
    }

    PyObject* toPy(const occa::memory &memory) {
      static PyTypeObject *Memory = getTypeFromModule("occa.c.memory", "Memory");
      return newCoreType(Memory, (void*) memory.getModeMemory());
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

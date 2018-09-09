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

#ifndef OCCA_PY_TYPES_HEADER
#define OCCA_PY_TYPES_HEADER

#include "defines.hpp"


namespace occa {
  namespace py {
    //---[ PyType ]---------------------
    typedef struct {
      PyObject_HEAD
      occa::device *device;
    } Device;

    typedef struct {
      PyObject_HEAD
      occa::memory *memory;
    } Memory;

    typedef struct {
      PyObject_HEAD
      occa::kernel *kernel;
    } Kernel;

    typedef struct {
      PyObject_HEAD
      occa::stream *stream;
    } Stream;

    typedef struct {
      PyObject_HEAD
      occa::streamTag *streamTag;
    } StreamTag;
    //==================================


    //---[ Type Getters ]---------------
    static PyTypeObject* getTypeFromModule(const std::string &moduleName,
                                           const std::string &className) {
      PyObject *module = PyImport_ImportModule(moduleName.c_str());
      PyTypeObject *type = (PyTypeObject*) PyObject_GetAttrString(module, className.c_str());
      Py_DECREF(module);
      return type;
    }

    PyTypeObject* ErrorType() {
      static PyTypeObject *Error = getTypeFromModule("occa.exceptions", "CError");
      return Error;
    }

    PyTypeObject* DeviceType() {
      static PyTypeObject *Device = getTypeFromModule("occa.c.device", "Device");
      return Device;
    }

    PyTypeObject* KernelType() {
      static PyTypeObject *Kernel = getTypeFromModule("occa.c.kernel", "Kernel");
      return Kernel;
    }

    PyTypeObject* MemoryType() {
      static PyTypeObject *Memory = getTypeFromModule("occa.c.memory", "Memory");
      return Memory;
    }

    PyTypeObject* StreamType() {
      static PyTypeObject *Stream = getTypeFromModule("occa.c.stream", "Stream");
      return Stream;
    }

    PyTypeObject* StreamTagType() {
      static PyTypeObject *StreamTag = getTypeFromModule("occa.c.streamtag", "StreamTag");
      return StreamTag;
    }
    //==================================


    //---[ Checks ]---------------------
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
    //==================================


    //---[ Py -> C ]--------------------
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
    //==================================


    //---[ Custom Types ]---------------
    class object {
    public:
      PyObject *obj;

      inline object(PyObject *obj_ = NULL) :
        obj(obj_) {
        if (obj) {
          Py_INCREF(obj);
        }
      }

      inline ~object() {
        if (obj) {
          Py_DECREF(obj);
        }
      }

      inline void setObj(PyObject *obj_) {
        if (obj) {
          Py_DECREF(obj);
        }
        obj = obj_;
        Py_INCREF(obj);
      }

      inline bool isInitialized() {
        return obj;
      }
    };

    class list : public object {
    public:
      inline list(PyObject *obj_ = NULL) :
        object(obj_) {}

      inline int size() {
        return (obj
                ? (int) PyList_Size(obj)
                : 0);
      }

      inline PyObject* operator [] (const int index) {
        return (obj
                ? PyList_GetItem(obj, index)
                : NULL);
      }
    };

    class ndArray : public object {
    public:
      inline ndArray(PyObject *obj_ = NULL) :
        object(obj_) {}

      inline void* ptr() {
        return (obj
                ? PyArray_DATA((PyArrayObject*) obj)
                : NULL);
      }

      inline int size() {
        PyArrayObject *arrayPtr = (PyArrayObject*) obj;
        return (arrayPtr
                ? PyArray_SIZE(arrayPtr) * PyArray_ITEMSIZE(arrayPtr)
                : 0);
      }
    };

    class memoryLike : public object {
    public:
      inline memoryLike(PyObject *obj_ = NULL) :
        object(obj_) {}

      inline bool isInitialized() {
        return obj;
      }

      inline bool isMemory() {
        return (obj && occa::py::isMemory(obj));
      }

      inline bool isNDArray() {
        return (obj && !occa::py::isMemory(obj));
      }

      inline occa::memory memory() {
        return *(((Memory*) obj)->memory);
      }

      inline char* ptr() {
        return (char*) occa::py::ptr(obj);
      }
    };
    //==================================
  }
}

#endif

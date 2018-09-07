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
    static PyTypeObject* getTypeFromModule(const std::string &moduleName,
                                           const std::string &className) {
      PyObject *module = PyImport_ImportModule(moduleName.c_str());
      return (PyTypeObject*) PyObject_GetAttrString(module, className.c_str());
    }

    PyTypeObject* ErrorType() {
      static PyTypeObject *Error = NULL;

      if (!Error) {
        Error = getTypeFromModule("occa.c.exception", "Error");

        PyObject *name = PyUnicode_FromString("occa.c.Error");
        if (name) {
          PyObject_SetAttrString((PyObject*) Error, "__name__", name);
        }
      }

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
  }
}

#endif

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

#ifndef OCCA_PY_UTILS_HEADER
#define OCCA_PY_UTILS_HEADER

#include "defines.hpp"


#define OCCA_TRY(...)                           \
  try {                                         \
    __VA_ARGS__                                 \
  } catch (occa::exception e) {                 \
    occa::py::raise(e);                         \
  }


namespace occa {
  namespace py {
    static PyObject *Error = NULL;

    static void raise(occa::exception e) {
      if (occa::py::Error == NULL) {
        PyObject *module = PyImport_ImportModule("occa.c.exception");
        occa::py::Error = PyObject_GetAttrString(module, "Error");
      }

      std::string message = e.message;
      message += '\n';
      message += e.location();
      PyErr_SetString(occa::py::Error, message.c_str());
    }
  }
}


#endif

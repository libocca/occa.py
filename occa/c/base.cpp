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
#include "header.hpp"


//---[ Globals & Flags ]----------------
static PyObject* py_occa_settings(PyObject *self) {
  return occa::py::toPy(
    occa::settings()
  );
}

static PyObject* py_occa_print_mode_info(PyObject *self) {
  occa::printModeInfo();
  return occa::py::None();
}
//======================================


//---[ Device ]-------------------------
static PyObject* py_occa_host(PyObject *self) {
  return occa::py::toPy(
    occa::host()
  );
}

static PyObject* py_occa_get_device(PyObject *self) {
  return occa::py::toPy(
    occa::getDevice()
  );
}

static PyObject* py_occa_set_device(PyObject *self,
                                    PyObject *args,
                                    PyObject *kwargs) {
  static const char *kwargNames[] = {
    "props", "device", NULL
  };

  char *info = NULL;
  PyObject *deviceObj = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sO", (char**) kwargNames,
                                   &info, &deviceObj)) {
    return NULL;
  }

  occa::device *device;

  OCCA_TRY(
    if (info) {
      device = new occa::device(info);
    } else {
      device = new occa::device(
        (occa::modeDevice_t*) occa::py::ptr(deviceObj)
      );
    }
  );

  return occa::py::toPy(*device);
}

static PyObject* py_occa_device_properties(PyObject *self) {
  return occa::py::toPy(
    occa::deviceProperties()
  );
}

static PyObject* py_occa_finish(PyObject *self) {
  occa::finish();
  return occa::py::None();
}

static PyObject* py_occa_create_stream(PyObject *self) {
  return occa::py::toPy(
    occa::createStream()
  );
}

static PyObject* py_occa_get_stream(PyObject *self) {
  return occa::py::toPy(
    occa::getStream()
  );
}

static PyObject* py_occa_set_stream(PyObject *self,
                                    PyObject *args,
                                    PyObject *kwargs) {
  static const char *kwargNames[] = {
    "stream", NULL
  };

  PyObject *streamObj = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**) kwargNames,
                                   &streamObj)) {
    return NULL;
  }

  occa::setStream(
    (occa::modeStream_t*) occa::py::ptr(streamObj)
  );

  return occa::py::None();
}

static PyObject* py_occa_tag_stream(PyObject *self) {
  return occa::py::toPy(
    occa::tagStream()
  );
}

static PyObject* py_occa_wait_for(PyObject *self,
                                  PyObject *args,
                                  PyObject *kwargs) {
  static const char *kwargNames[] = {
    "tag", NULL
  };

  PyObject *tagObj = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**) kwargNames,
                                   &tagObj)) {
    return NULL;
  }

  occa::waitFor(
    (occa::modeStreamTag_t*) occa::py::ptr(tagObj)
  );
  return occa::py::None();
}

static PyObject* py_occa_time_between(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  static const char *kwargNames[] = {
    "start", "end", NULL
  };

  PyObject *startObj = NULL;
  PyObject *endObj = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char**) kwargNames,
                                   &startObj, &endObj)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::timeBetween(
      (occa::modeStreamTag_t*) occa::py::ptr(startObj),
      (occa::modeStreamTag_t*) occa::py::ptr(endObj)
    )
  );
}
//======================================


//---[ Kernel ]-------------------------
static PyObject* py_occa_build_kernel(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  static const char *kwargNames[] = {
    "filename", "kernel", "props", NULL
  };

  char *filename = NULL;
  char *kernel = NULL;
  char *propsStr = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sss", (char**) kwargNames,
                                   &filename, &kernel, &propsStr)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::buildKernel(filename,
                      kernel,
                      occa::properties(propsStr))
  );
}

static PyObject* py_occa_build_kernel_from_string(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  static const char *kwargNames[] = {
    "source", "kernel", "props", NULL
  };

  char *source = NULL;
  char *kernel = NULL;
  char *propsStr = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sss", (char**) kwargNames,
                                   &source, &kernel, &propsStr)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::buildKernelFromString(source,
                                kernel,
                                occa::properties(propsStr))
  );
}

static PyObject* py_occa_build_kernel_from_binary(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  static const char *kwargNames[] = {
    "filename", "kernel", "props", NULL
  };

  char *filename = NULL;
  char *kernel = NULL;
  char *propsStr = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sss", (char**) kwargNames,
                                   &filename, &kernel, &propsStr)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::buildKernelFromBinary(filename,
                                kernel,
                                occa::properties(propsStr))
  );
}
//======================================


//---[ Memory ]-------------------------
static PyObject* py_occa_malloc(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs) {
  static const char *kwargNames[] = {
    "bytes", "src", "props", NULL
  };

  // TODO: Swap src with numpy arrays or memory objects

  long long bytes = -1;
  PyObject *src = NULL;
  char *propsStr = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "L|Os", (char**) kwargNames,
                                   &bytes, &src, &propsStr)) {
    return NULL;
  }

  occa::properties props;
  if (propsStr) {
    props = occa::properties(propsStr);
  }

  return occa::py::toPy(
    occa::malloc(bytes, NULL, props)
  );
}

static PyObject* py_occa_copy_mem_to_mem(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  static const char *kwargNames[] = {
    "dest", "src",
    "bytes", "destOffset", "srcOffset",
    "props", NULL
  };

  PyObject *destObj = NULL;
  PyObject *srcObj = NULL;
  long long bytes = -1;
  long long destOffset = -1;
  long long srcOffset = -1;
  PyObject *propsObj = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOLLLO", (char**) kwargNames,
                                   &destObj, &srcObj, &bytes, &destOffset, &srcOffset, &propsObj)) {
    return NULL;
  }

  return occa::py::None();
}

static PyObject* py_occa_copy_ptr_to_mem(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  static const char *kwargNames[] = {
    "dest", "src",
    "bytes", "destOffset", "srcOffset",
    "props", NULL
  };

  // TODO: Swap src with numpy arrays

  PyObject *dest = NULL;
  PyObject *src = NULL;
  long long bytes = -1;
  long long destOffset = -1;
  long long srcOffset = -1;
  PyObject *props = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOLLLO", (char**) kwargNames,
                                   &dest, &src, &bytes, &destOffset, &srcOffset, &props)) {
    return NULL;
  }

  return occa::py::None();
}

static PyObject* py_occa_copy_mem_to_ptr(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  static const char *kwargNames[] = {
    "dest", "src",
    "bytes", "destOffset", "srcOffset",
    "props", NULL
  };

  // TODO: Swap dest with numpy arrays

  PyObject *dest = NULL;
  PyObject *src = NULL;
  long long bytes = -1;
  long long destOffset = -1;
  long long srcOffset = -1;
  PyObject *props = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOLLLO", (char**) kwargNames,
                                   &dest, &src, &bytes, &destOffset, &srcOffset, &props)) {
    return NULL;
  }

  return occa::py::None();
}
//======================================
#define BASE_METHOD_NO_ARGS(FUNC)               \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, py_occa_##FUNC)

#define BASE_METHOD_WITH_KWARGS(FUNC)               \
  OCCA_PY_METHOD_WITH_KWARGS(#FUNC, py_occa_##FUNC)


static bool base_has_valid_module() {
  return true;
}

static void base_init_module(PyObject *module) {}

OCCA_PY_MODULE(
  base,
  BASE_METHOD_NO_ARGS(settings),
  BASE_METHOD_NO_ARGS(print_mode_info),
  BASE_METHOD_NO_ARGS(host),
  BASE_METHOD_NO_ARGS(get_device),
  BASE_METHOD_WITH_KWARGS(set_device),
  BASE_METHOD_NO_ARGS(device_properties),
  BASE_METHOD_NO_ARGS(finish),
  BASE_METHOD_NO_ARGS(create_stream),
  BASE_METHOD_NO_ARGS(get_stream),
  BASE_METHOD_WITH_KWARGS(set_stream),
  BASE_METHOD_NO_ARGS(tag_stream),
  BASE_METHOD_WITH_KWARGS(wait_for),
  BASE_METHOD_WITH_KWARGS(time_between),
  BASE_METHOD_WITH_KWARGS(build_kernel),
  BASE_METHOD_WITH_KWARGS(build_kernel_from_string),
  BASE_METHOD_WITH_KWARGS(build_kernel_from_binary),
  BASE_METHOD_WITH_KWARGS(malloc),
  BASE_METHOD_WITH_KWARGS(copy_mem_to_mem),
  BASE_METHOD_WITH_KWARGS(copy_ptr_to_mem),
  BASE_METHOD_WITH_KWARGS(copy_mem_to_ptr)
);

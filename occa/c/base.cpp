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
  occa::device device;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("device", device)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  if (device.isInitialized()) {
    occa::setDevice(device);
  } else {
    OCCA_TRY(
      occa::setDevice(props);
    );
  }

  return occa::py::None();
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
  occa::stream stream;

  occa::py::kwargParser parser;
  parser
    .add("stream", stream);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  occa::setStream(stream);

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
  occa::streamTag tag;

  occa::py::kwargParser parser;
  parser
    .add("tag", tag);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  occa::waitFor(tag);
  return occa::py::None();
}

static PyObject* py_occa_time_between(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  occa::streamTag start, end;

  occa::py::kwargParser parser;
  parser
    .add("start", start)
    .add("end", end);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::timeBetween(start, end)
  );
}
//======================================


//---[ Kernel ]-------------------------
static PyObject* py_occa_build_kernel(PyObject *self,
                                      PyObject *args,
                                      PyObject *kwargs) {
  std::string filename, kernel;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .add("filename", filename)
    .add("kernel", kernel)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::buildKernel(filename, kernel, props)
  );
}

static PyObject* py_occa_build_kernel_from_string(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  std::string source, kernel;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .add("source", source)
    .add("kernel", kernel)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::buildKernelFromString(source, kernel, props)
  );
}

static PyObject* py_occa_build_kernel_from_binary(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  std::string filename, kernel;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .add("filename", filename)
    .add("kernel", kernel)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::buildKernelFromBinary(filename, kernel, props)
  );
}
//======================================


//---[ Memory ]-------------------------
static PyObject* py_occa_malloc(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs) {
  long long bytes = -1;
  void *src = NULL;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .add("bytes", bytes)
    .add("src", src)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    occa::malloc(bytes, src, props)
  );
}

static PyObject* py_occa_memcpy(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs) {
  occa::py::memoryLike dest;
  occa::py::memoryLike src;
  long long bytes = -1;
  long long destOffset = 0;
  long long srcOffset = 0;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .add("dest", dest)
    .add("src", src)
    .startOptionalKwargs()
    .add("bytes", bytes)
    .add("dest_offset", destOffset)
    .add("src_offset", srcOffset)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  if (!dest.isInitialized()) {
    occa::py::raise("dest is not an occa.Memory or numpy.ndarray object");
    return NULL;
  }
  if (!src.isInitialized()) {
    occa::py::raise("src is not an occa.Memory or numpy.ndarray object");
    return NULL;
  }

  OCCA_TRY(
    if (dest.isMemory()) {
      if (src.isMemory()) {
        occa::memcpy(dest.memory(),
                     src.memory(),
                     bytes,
                     destOffset,
                     srcOffset,
                     props);
      } else {
        occa::memcpy(dest.memory(),
                     (void*) (src.ptr() + srcOffset),
                     bytes,
                     destOffset,
                     props);
      }
    } else {
      if (src.isMemory()) {
        occa::memcpy((void*) (dest.ptr() + destOffset),
                     src.memory(),
                     bytes,
                     srcOffset,
                     props);
      } else {
        occa::memcpy((void*) (dest.ptr() + destOffset),
                     (void*) (src.ptr() + srcOffset),
                     bytes,
                     props);
      }
    }
  );

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
  BASE_METHOD_WITH_KWARGS(memcpy)
);

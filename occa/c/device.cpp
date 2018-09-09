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


//---[ Class Python Methods ]-----------
static int Device_init(occa::py::Device *self,
                       PyObject *args,
                       PyObject *kwargs) {
  self->device = NULL;

  occa::device device;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("device", device)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return -1;
  }

  if (device.isInitialized()) {
    self->device = new occa::device(device);
  } else if (props.isInitialized()) {
    OCCA_INIT_TRY(
      self->device = new occa::device(props);
    );
  }

  return 0;
}

static void Device_dealloc(occa::py::Device *self) {
  delete self->device;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* Device_is_initialized(occa::py::Device *self) {
  return occa::py::toPy(
    (bool) (self->device &&
            self->device->isInitialized())
  );
}

static PyObject* Device_free(occa::py::Device *self) {
  if (self->device) {
    self->device->free();
  }
  return occa::py::None();
}

static PyObject* Device_mode(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->mode()
  );
}

static PyObject* Device_properties(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->properties()
  );
}

static PyObject* Device_kernel_properties(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->kernelProperties()
  );
}

static PyObject* Device_memory_properties(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->memoryProperties()
  );
}

static PyObject* Device_memory_size(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    (long long) self->device->memorySize()
  );
}

static PyObject* Device_memory_allocated(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    (long long) self->device->memoryAllocated()
  );
}

static PyObject* Device_finish(occa::py::Device *self) {
  if (self->device) {
    self->device->finish();
  }
  return occa::py::None();
}

static PyObject* Device_has_separate_memory_space(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->hasSeparateMemorySpace()
  );
}


//  |---[ Stream ]----------------------
static PyObject* Device_create_stream(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->createStream()
  );
}

static PyObject* Device_get_stream(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->getStream()
  );
}

static PyObject* Device_set_stream(occa::py::Device *self,
                                   PyObject *args,
                                   PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

  occa::stream stream;

  occa::py::kwargParser parser;
  parser
    .add("stream", stream);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  self->device->setStream(stream);

  return occa::py::None();
}

static PyObject* Device_tag_stream(occa::py::Device *self) {
  if (!self->device) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->device->tagStream()
  );
}

static PyObject* Device_wait_for(occa::py::Device *self,
                                 PyObject *args,
                                 PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

  occa::streamTag tag;

  occa::py::kwargParser parser;
  parser
    .add("tag", tag);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  self->device->waitFor(tag);
  return occa::py::None();
}

static PyObject* Device_time_between(occa::py::Device *self,
                                     PyObject *args,
                                     PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

  occa::streamTag start, end;

  occa::py::kwargParser parser;
  parser
    .add("start", start)
    .add("end", end);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    self->device->timeBetween(start, end)
  );
}
//  |===================================


//  |---[ Kernel ]----------------------
static PyObject* Device_build_kernel(occa::py::Device *self,
                                     PyObject *args,
                                     PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

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
    self->device->buildKernel(filename, kernel, props)
  );
}

static PyObject* Device_build_kernel_from_string(occa::py::Device *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

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
    self->device->buildKernelFromString(source, kernel, props)
  );
}

static PyObject* Device_build_kernel_from_binary(occa::py::Device *self,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

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
    self->device->buildKernelFromBinary(filename, kernel, props)
  );
}
//  |===================================


//  |---[ Memory ]----------------------
static PyObject* Device_malloc(occa::py::Device *self,
                               PyObject *args,
                               PyObject *kwargs) {
  if (!self->device) {
    return occa::py::None();
  }

  long long bytes = 0;
  occa::py::ndArray src;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("bytes", bytes)
    .add("src", src)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  if (!bytes) {
    bytes = src.size();
  }

  return occa::py::toPy(
    self->device->malloc(bytes,
                         src.ptr(),
                         props)
  );
}
//  |===================================
//======================================


//---[ Module ]-------------------------
#define DEVICE_METHOD_NO_ARGS(FUNC)             \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, Device_##FUNC)

#define DEVICE_METHOD_WITH_KWARGS(FUNC)             \
  OCCA_PY_METHOD_WITH_KWARGS(#FUNC, Device_##FUNC)

OCCA_PY_METHODS(
  Device_methods,
  DEVICE_METHOD_NO_ARGS(is_initialized),
  DEVICE_METHOD_NO_ARGS(free),
  DEVICE_METHOD_NO_ARGS(mode),
  DEVICE_METHOD_NO_ARGS(properties),
  DEVICE_METHOD_NO_ARGS(kernel_properties),
  DEVICE_METHOD_NO_ARGS(memory_properties),
  DEVICE_METHOD_NO_ARGS(memory_size),
  DEVICE_METHOD_NO_ARGS(memory_allocated),
  DEVICE_METHOD_NO_ARGS(finish),
  DEVICE_METHOD_NO_ARGS(has_separate_memory_space),
  DEVICE_METHOD_NO_ARGS(create_stream),
  DEVICE_METHOD_NO_ARGS(get_stream),
  DEVICE_METHOD_WITH_KWARGS(set_stream),
  DEVICE_METHOD_NO_ARGS(tag_stream),
  DEVICE_METHOD_WITH_KWARGS(wait_for),
  DEVICE_METHOD_WITH_KWARGS(time_between),
  DEVICE_METHOD_WITH_KWARGS(build_kernel),
  DEVICE_METHOD_WITH_KWARGS(build_kernel_from_string),
  DEVICE_METHOD_WITH_KWARGS(build_kernel_from_binary),
  DEVICE_METHOD_WITH_KWARGS(malloc)
);

static PyTypeObject DeviceType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.Device",                          // tp_name
  sizeof(occa::py::Device),                 // tp_basicsize
  0,                                        // tp_itemsize
  (destructor) Device_dealloc,              // tp_dealloc
  0,                                        // tp_print
  0,                                        // tp_getattr
  0,                                        // tp_setattr
  0,                                        // tp_reserved
  0,                                        // tp_repr
  0,                                        // tp_as_number
  0,                                        // tp_as_sequence
  0,                                        // tp_as_mapping
  0,                                        // tp_hash
  0,                                        // tp_call
  0,                                        // tp_str
  0,                                        // tp_getattro
  0,                                        // tp_setattro
  0,                                        // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
  "Wrapper for occa::device",               // tp_doc
  0,                                        // tp_traverse
  0,                                        // tp_clear
  0,                                        // tp_richcompare
  0,                                        // tp_weaklistoffset
  0,                                        // tp_iter
  0,                                        // tp_iternext
  Device_methods,                           // tp_methods
  0,                                        // tp_members
  0,                                        // tp_getset
  0,                                        // tp_base
  0,                                        // tp_dict
  0,                                        // tp_descr_get
  0,                                        // tp_descr_set
  0,                                        // tp_dictoffset
  (initproc) Device_init,                   // tp_init
  0,                                        // tp_alloc
  0                                         // tp_new
};

static bool device_has_valid_module() {
  DeviceType.tp_new = PyType_GenericNew;
  return PyType_Ready(&DeviceType) >= 0;
}

static void device_init_module(PyObject *module) {
  Py_INCREF(&DeviceType);
  PyModule_AddObject(module,
                     "Device",
                     (PyObject*) &DeviceType);
}

OCCA_PY_MODULE(device, OCCA_PY_NO_METHODS)
//======================================

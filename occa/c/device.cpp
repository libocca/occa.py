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
typedef struct {
  PyObject_HEAD
  occa::device *device;
} Device;

static PyObject* Device_new(PyTypeObject *type,
                            PyObject *args,
                            PyObject *kwargs) {
  Device *self = (Device*) type->tp_alloc(type, 0);
  if (self != NULL) {
    self->device = NULL;
  }
  return (PyObject*) self;
}

static int Device_init(Device *self,
                       PyObject *args,
                       PyObject *kwargs) {
  const char *info = NULL;
  if (!PyArg_ParseTuple(args, "|s", &info)) {
    return -1;
  }

  delete self->device;
  self->device = NULL;
  if (info) {
    OCCA_TRY_AND_RETURNS(
      -1,
      self->device = new occa::device(info);
    );
  }

  return 0;
}

static void Device_dealloc(Device *self) {
  delete self->device;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* Device_is_initialized(Device *self,
                                       PyObject *args) {
  if (self->device &&
      self->device->isInitialized()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* Device_mode(Device *self,
                             PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  return occa::py::toPy(self->device->mode());
}

static PyObject* Device_properties(Device *self,
                                   PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  return occa::py::toPy(self->device->properties());
}

static PyObject* Device_kernel_properties(Device *self,
                                          PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  return occa::py::toPy(self->device->kernelProperties());
}

static PyObject* Device_memory_properties(Device *self,
                                          PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  return occa::py::toPy(self->device->memoryProperties());
}

static PyObject* Device_memory_size(Device *self,
                                    PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_memory_allocated(Device *self,
                                         PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_finish(Device *self,
                               PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_has_separate_memory_space(Device *self,
                                                  PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}


//  |---[ Stream ]----------------------
static PyObject* Device_create_stream(Device *self,
                                      PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_get_stream(Device *self,
                                   PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_set_stream(Device *self,
                                   PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_tag_stream(Device *self,
                                   PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_wait_for_tag(Device *self,
                                     PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_time_between_tags(Device *self,
                                          PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}
//  |===================================


//  |---[ Kernel ]----------------------
static PyObject* Device_build_kernel(Device *self,
                                     PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_build_kernel_from_string(Device *self,
                                                 PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_build_kernel_from_binary(Device *self,
                                                 PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}
//  |===================================


//  |---[ Memory ]----------------------
static PyObject* Device_malloc(Device *self,
                               PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}

static PyObject* Device_umalloc(Device *self,
                                PyObject *args) {
  if (self->device == NULL) {
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
}
//  |===================================
//======================================


//---[ Module ]-------------------------
#define DEVICE_METHOD(FUNC) OCCA_PY_METHOD(#FUNC, Device_##FUNC)

OCCA_PY_METHODS(
  Device_methods,
  DEVICE_METHOD(is_initialized),
  DEVICE_METHOD(mode),
  DEVICE_METHOD(properties),
  DEVICE_METHOD(kernel_properties),
  DEVICE_METHOD(memory_properties),
  DEVICE_METHOD(memory_size),
  DEVICE_METHOD(memory_allocated),
  DEVICE_METHOD(finish),
  DEVICE_METHOD(has_separate_memory_space),
  DEVICE_METHOD(create_stream),
  DEVICE_METHOD(get_stream),
  DEVICE_METHOD(set_stream),
  DEVICE_METHOD(tag_stream),
  DEVICE_METHOD(wait_for_tag),
  DEVICE_METHOD(time_between_tags),
  DEVICE_METHOD(build_kernel),
  DEVICE_METHOD(build_kernel_from_string),
  DEVICE_METHOD(build_kernel_from_binary),
  DEVICE_METHOD(malloc),
  DEVICE_METHOD(umalloc)
);

static PyTypeObject DeviceType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.Device",             // tp_name
  sizeof(Device),              // tp_basicsize
  0,                           // tp_itemsize
  (destructor) Device_dealloc, // tp_dealloc
  0,                           // tp_print
  0,                           // tp_getattr
  0,                           // tp_setattr
  0,                           // tp_reserved
  0,                           // tp_repr
  0,                           // tp_as_number
  0,                           // tp_as_sequence
  0,                           // tp_as_mapping
  0,                           // tp_hash
  0,                           // tp_call
  0,                           // tp_str
  0,                           // tp_getattro
  0,                           // tp_setattro
  0,                           // tp_as_buffer
  Py_TPFLAGS_DEFAULT,          // tp_flags
  "Wrapper for occa::device",  // tp_doc
  0,                           // tp_traverse
  0,                           // tp_clear
  0,                           // tp_richcompare
  0,                           // tp_weaklistoffset
  0,                           // tp_iter
  0,                           // tp_iternext
  Device_methods,              // tp_methods
  0,                           // tp_members
  0,                           // tp_getset
  0,                           // tp_base
  0,                           // tp_dict
  0,                           // tp_descr_get
  0,                           // tp_descr_set
  0,                           // tp_dictoffset
  (initproc) Device_init,      // tp_init
  0,                           // tp_alloc
  Device_new,                  // tp_new
};

static bool device_has_valid_module() {
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

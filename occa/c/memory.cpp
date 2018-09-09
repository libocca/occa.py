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
static int Memory_init(occa::py::Memory *self,
                       PyObject *args,
                       PyObject *kwargs) {
  self->memory = NULL;

  occa::memory memory;
  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("memory", memory);

  if (!parser.parse(args, kwargs)) {
    return -1;
  }

  if (memory.isInitialized()) {
    self->memory = new occa::memory(memory);
  }

  return 0;
}

static void Memory_dealloc(occa::py::Memory *self) {
  delete self->memory;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* Memory_is_initialized(occa::py::Memory *self) {
  return occa::py::toPy(
    (bool) (self->memory &&
            self->memory->isInitialized())
  );
}

static PyObject* Memory_free(occa::py::Memory *self) {
  if (self->memory) {
    self->memory->free();
  }
  return occa::py::None();
}

static PyObject* Memory_mode(occa::py::Memory *self) {
  if (!self->memory) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->memory->mode()
  );
}

static PyObject* Memory_properties(occa::py::Memory *self) {
  if (!self->memory) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->memory->properties()
  );
}

static PyObject* Memory_get_device(occa::py::Memory *self) {
  if (!self->memory) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->memory->getDevice()
  );
}

static PyObject* Memory_size(occa::py::Memory *self) {
  if (!self->memory) {
    return occa::py::None();
  }

  return occa::py::toPy(
    (long long) self->memory->size()
  );
}

static PyObject* Memory_slice(occa::py::Memory *self,
                              PyObject *args,
                              PyObject *kwargs) {
  if (!self->memory) {
    return occa::py::None();
  }

  long long offset = 0;
  long long bytes = -1;

  occa::py::kwargParser parser;
  parser
    .add("offset", offset)
    .add("bytes", bytes);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  return occa::py::toPy(
    self->memory->slice(offset, bytes)
  );
}

//---[ UVA ]----------------------------
static PyObject* Memory_is_managed(occa::py::Memory *self) {
  return occa::py::toPy(
    (bool) (self->memory &&
            self->memory->isManaged())
  );
}

static PyObject* Memory_in_device(occa::py::Memory *self) {
  return occa::py::toPy(
    (bool) (self->memory &&
            self->memory->inDevice())
  );
}

static PyObject* Memory_is_stale(occa::py::Memory *self) {
  return occa::py::toPy(
    (bool) (self->memory &&
            self->memory->isStale())
  );
}

static PyObject* Memory_start_managing(occa::py::Memory *self) {
  if (self->memory) {
    self->memory->startManaging();
  }
  return occa::py::None();
}

static PyObject* Memory_stop_managing(occa::py::Memory *self) {
  if (self->memory) {
    self->memory->stopManaging();
  }
  return occa::py::None();
}

static PyObject* Memory_sync_to_device(occa::py::Memory *self,
                                       PyObject *args,
                                       PyObject *kwargs) {
  if (!self->memory) {
    return occa::py::None();
  }

  long long bytes = -1;
  long long offset = 0;

  occa::py::kwargParser parser;
  parser
    .add("bytes", bytes)
    .add("offset", offset);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  self->memory->syncToDevice(bytes, offset);

  return occa::py::None();
}

static PyObject* Memory_sync_to_host(occa::py::Memory *self,
                                     PyObject *args,
                                     PyObject *kwargs) {
  if (!self->memory) {
    return occa::py::None();
  }

  long long bytes = -1;
  long long offset = 0;

  occa::py::kwargParser parser;
  parser
    .add("bytes", bytes)
    .add("offset", offset);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  self->memory->syncToHost(bytes, offset);

  return occa::py::None();
}
//======================================

static PyObject* Memory_clone(occa::py::Memory *self) {
  if (!self->memory) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->memory->clone()
  );
}
//======================================


//---[ Module ]-------------------------
#define MEMORY_METHOD_NO_ARGS(FUNC)             \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, Memory_##FUNC)

#define MEMORY_METHOD_WITH_KWARGS(FUNC)             \
  OCCA_PY_METHOD_WITH_KWARGS(#FUNC, Memory_##FUNC)

OCCA_PY_METHODS(
  Memory_methods,
  MEMORY_METHOD_NO_ARGS(is_initialized),
  MEMORY_METHOD_NO_ARGS(free),
  MEMORY_METHOD_NO_ARGS(mode),
  MEMORY_METHOD_NO_ARGS(get_device),
  MEMORY_METHOD_NO_ARGS(properties),
  MEMORY_METHOD_NO_ARGS(size),
  MEMORY_METHOD_WITH_KWARGS(slice),
  MEMORY_METHOD_NO_ARGS(is_managed),
  MEMORY_METHOD_NO_ARGS(in_device),
  MEMORY_METHOD_NO_ARGS(is_stale),
  MEMORY_METHOD_NO_ARGS(start_managing),
  MEMORY_METHOD_NO_ARGS(stop_managing),
  MEMORY_METHOD_WITH_KWARGS(sync_to_device),
  MEMORY_METHOD_WITH_KWARGS(sync_to_host),
  MEMORY_METHOD_NO_ARGS(clone)
);

static PyTypeObject MemoryType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.Memory",                          // tp_name
  sizeof(occa::py::Memory),                 // tp_basicsize
  0,                                        // tp_itemsize
  (destructor) Memory_dealloc,              // tp_dealloc
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
  "Wrapper for occa::memory",               // tp_doc
  0,                                        // tp_traverse
  0,                                        // tp_clear
  0,                                        // tp_richcompare
  0,                                        // tp_weaklistoffset
  0,                                        // tp_iter
  0,                                        // tp_iternext
  Memory_methods,                           // tp_methods
  0,                                        // tp_members
  0,                                        // tp_getset
  0,                                        // tp_base
  0,                                        // tp_dict
  0,                                        // tp_descr_get
  0,                                        // tp_descr_set
  0,                                        // tp_dictoffset
  (initproc) Memory_init,                   // tp_init
  0,                                        // tp_alloc
  0                                         // tp_new
};

static bool memory_has_valid_module() {
  MemoryType.tp_new = PyType_GenericNew;
  return PyType_Ready(&MemoryType) >= 0;
}

static void memory_init_module(PyObject *module) {
  Py_INCREF(&MemoryType);
  PyModule_AddObject(module,
                     "Memory",
                     (PyObject*) &MemoryType);
}

OCCA_PY_MODULE(memory, OCCA_PY_NO_METHODS)
//======================================

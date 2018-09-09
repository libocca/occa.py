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
static int Stream_init(occa::py::Stream *self,
                       PyObject *args,
                       PyObject *kwargs) {
  self->stream = NULL;

  occa::stream stream;
  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("stream", stream);

  if (!parser.parse(args, kwargs)) {
    return -1;
  }

  if (stream.isInitialized()) {
    self->stream = new occa::stream(stream);
  }

  return 0;
}

static void Stream_dealloc(occa::py::Stream *self) {
  delete self->stream;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* Stream_is_initialized(occa::py::Stream *self) {
  return occa::py::toPy(
    (bool) (self->stream &&
            self->stream->isInitialized())
  );
}

static PyObject* Stream_free(occa::py::Stream *self) {
  if (self->stream) {
    self->stream->free();
  }
  return occa::py::None();
}

static PyObject* Stream_mode(occa::py::Stream *self) {
  if (!self->stream) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->stream->mode()
  );
}

static PyObject* Stream_properties(occa::py::Stream *self) {
  if (!self->stream) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->stream->properties()
  );
}

static PyObject* Stream_get_device(occa::py::Stream *self) {
  if (!self->stream) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->stream->getDevice()
  );
}
//======================================


//---[ Module ]-------------------------
#define STREAM_METHOD_NO_ARGS(FUNC)             \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, Stream_##FUNC)

OCCA_PY_METHODS(
  Stream_methods,
  STREAM_METHOD_NO_ARGS(is_initialized),
  STREAM_METHOD_NO_ARGS(free),
  STREAM_METHOD_NO_ARGS(mode),
  STREAM_METHOD_NO_ARGS(properties),
  STREAM_METHOD_NO_ARGS(get_device)
);

static PyTypeObject StreamType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.Stream",                          // tp_name
  sizeof(occa::py::Stream),                 // tp_basicsize
  0,                                        // tp_itemsize
  (destructor) Stream_dealloc,              // tp_dealloc
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
  "Wrapper for occa::stream",               // tp_doc
  0,                                        // tp_traverse
  0,                                        // tp_clear
  0,                                        // tp_richcompare
  0,                                        // tp_weaklistoffset
  0,                                        // tp_iter
  0,                                        // tp_iternext
  Stream_methods,                           // tp_methods
  0,                                        // tp_members
  0,                                        // tp_getset
  0,                                        // tp_base
  0,                                        // tp_dict
  0,                                        // tp_descr_get
  0,                                        // tp_descr_set
  0,                                        // tp_dictoffset
  (initproc) Stream_init,                   // tp_init
  0,                                        // tp_alloc
  0                                         // tp_new
};

static bool stream_has_valid_module() {
  StreamType.tp_new = PyType_GenericNew;
  return PyType_Ready(&StreamType) >= 0;
}

static void stream_init_module(PyObject *module) {
  Py_INCREF(&StreamType);
  PyModule_AddObject(module,
                     "Stream",
                     (PyObject*) &StreamType);
}

OCCA_PY_MODULE(stream, OCCA_PY_NO_METHODS)
//======================================

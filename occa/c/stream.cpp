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

  OCCA_INIT_TRY(
    if (stream.isInitialized()) {
      self->stream = new occa::stream(stream);
    }
  );

  return 0;
}

static void Stream_dealloc(occa::py::Stream *self) {
  delete self->stream;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* Stream_is_initialized(occa::py::Stream *self) {
  OCCA_TRY(
    return occa::py::toPy(
      (bool) (self->stream &&
              self->stream->isInitialized())
    );
  );
}

static PyObject* Stream_free(occa::py::Stream *self) {
  OCCA_TRY(
    if (self->stream) {
      self->stream->free();
    }
  );
  return occa::py::None();
}

static PyObject* Stream_mode(occa::py::Stream *self) {
  if (!self->stream) {
    return occa::py::None();
  }
  OCCA_TRY(
    return occa::py::toPy(
      self->stream->mode()
    );
  );
}

static PyObject* Stream_properties(occa::py::Stream *self) {
  if (!self->stream) {
    return occa::py::None();
  }
  OCCA_TRY(
    return occa::py::toPy(
      self->stream->properties()
    );
  );
}

static PyObject* Stream_get_device(occa::py::Stream *self) {
  if (!self->stream) {
    return occa::py::None();
  }
  OCCA_TRY(
    return occa::py::toPy(
      self->stream->getDevice()
    );
  );
}

static PyObject* Stream_ptr_as_long(occa::py::Stream *self) {
  OCCA_TRY(
    return occa::py::toPy(
      (long long) self->stream->getModeStream()
    );
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
  STREAM_METHOD_NO_ARGS(get_device),
  STREAM_METHOD_NO_ARGS(ptr_as_long)
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

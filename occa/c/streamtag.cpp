#include "header.hpp"


//---[ Class Python Methods ]-----------
static int StreamTag_init(occa::py::StreamTag *self,
                          PyObject *args,
                          PyObject *kwargs) {
  self->streamTag = NULL;

  occa::streamTag streamtag;
  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("streamtag", streamtag);

  if (!parser.parse(args, kwargs)) {
    return -1;
  }

  OCCA_INIT_TRY(
    if (streamtag.isInitialized()) {
      self->streamTag = new occa::streamTag(streamtag);
    }
  );

  return 0;
}

static void StreamTag_dealloc(occa::py::StreamTag *self) {
  delete self->streamTag;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* StreamTag_is_initialized(occa::py::StreamTag *self) {
  OCCA_TRY(
    return occa::py::toPy(
      (bool) (self->streamTag &&
              self->streamTag->isInitialized())
    );
  );
}

static PyObject* StreamTag_free(occa::py::StreamTag *self) {
  OCCA_TRY(
    if (self->streamTag) {
      self->streamTag->free();
    }
  );
  return occa::py::None();
}

static PyObject* StreamTag_get_device(occa::py::StreamTag *self) {
  if (!self->streamTag) {
    return occa::py::None();
  }
  OCCA_TRY(
    return occa::py::toPy(
      self->streamTag->getDevice()
    );
  );
}

static PyObject* StreamTag_wait(occa::py::StreamTag *self) {
  OCCA_TRY(
    if (self->streamTag) {
      self->streamTag->wait();
    }
  );
  return occa::py::None();
}

static PyObject* StreamTag_ptr_as_long(occa::py::StreamTag *self) {
  OCCA_TRY(
    return occa::py::toPy(
      (long long) self->streamTag->getModeStreamTag()
    );
  );
}
//======================================


//---[ Module ]-------------------------
#define STREAMTAG_METHOD_NO_ARGS(FUNC)            \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, StreamTag_##FUNC)

OCCA_PY_METHODS(
  StreamTag_methods,
  STREAMTAG_METHOD_NO_ARGS(is_initialized),
  STREAMTAG_METHOD_NO_ARGS(free),
  STREAMTAG_METHOD_NO_ARGS(get_device),
  STREAMTAG_METHOD_NO_ARGS(wait),
  STREAMTAG_METHOD_NO_ARGS(ptr_as_long)
);

static PyTypeObject StreamTagType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.StreamTag",                       // tp_name
  sizeof(occa::py::StreamTag),              // tp_basicsize
  0,                                        // tp_itemsize
  (destructor) StreamTag_dealloc,           // tp_dealloc
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
  "Wrapper for occa::streamTag",            // tp_doc
  0,                                        // tp_traverse
  0,                                        // tp_clear
  0,                                        // tp_richcompare
  0,                                        // tp_weaklistoffset
  0,                                        // tp_iter
  0,                                        // tp_iternext
  StreamTag_methods,                        // tp_methods
  0,                                        // tp_members
  0,                                        // tp_getset
  0,                                        // tp_base
  0,                                        // tp_dict
  0,                                        // tp_descr_get
  0,                                        // tp_descr_set
  0,                                        // tp_dictoffset
  (initproc) StreamTag_init,                // tp_init
  0,                                        // tp_alloc
  0                                         // tp_new
};

static bool streamtag_has_valid_module() {
  StreamTagType.tp_new = PyType_GenericNew;
  return PyType_Ready(&StreamTagType) >= 0;
}

static void streamtag_init_module(PyObject *module) {
  Py_INCREF(&StreamTagType);
  PyModule_AddObject(module,
                     "StreamTag",
                     (PyObject*) &StreamTagType);
}

OCCA_PY_MODULE(streamtag, OCCA_PY_NO_METHODS)
//======================================

#include "header.hpp"


//---[ Class Python Methods ]-----------
static int dtype_init(occa::py::dtype *self,
                       PyObject *args,
                       PyObject *kwargs) {
  self->dtype = NULL;

  occa::json json;
  std::string builtin;

  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("json", json)
    .add("builtin", builtin);

  if (!parser.parse(args, kwargs)) {
    return -1;
  }

  const bool hasJson   = json.isInitialized();
  const bool isBuiltin = (bool) builtin.size();

  if (!hasJson && !isBuiltin) {
    return -1;
  }

  OCCA_INIT_TRY(
    if (hasJson) {
      occa::dtype_t dtype = occa::dtype_t::fromJson(json);
      self->dtype = new occa::dtype_t(dtype);
      if (!dtype.isRegistered()) {
        self->dtype->registerType();
      }
    } else {
      self->dtype = new occa::dtype_t(
        occa::dtype_t::getBuiltin(builtin)
      );
    }
  );

  return 0;
}

static void dtype_dealloc(occa::py::dtype *self) {
  delete self->dtype;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Module ]-------------------------
#define DTYPE_METHOD_NO_ARGS(FUNC)             \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, dtype_##FUNC)

#define DTYPE_METHOD_WITH_KWARGS(FUNC)             \
  OCCA_PY_METHOD_WITH_KWARGS(#FUNC, dtype_##FUNC)

OCCA_PY_METHODS(
  dtype_methods,
  OCCA_PY_NO_METHODS
);

static PyTypeObject dtypeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.dtype",                           // tp_name
  sizeof(occa::py::dtype),                  // tp_basicsize
  0,                                        // tp_itemsize
  (destructor) dtype_dealloc,               // tp_dealloc
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
  "Wrapper for occa::dtype",                // tp_doc
  0,                                        // tp_traverse
  0,                                        // tp_clear
  0,                                        // tp_richcompare
  0,                                        // tp_weaklistoffset
  0,                                        // tp_iter
  0,                                        // tp_iternext
  dtype_methods,                            // tp_methods
  0,                                        // tp_members
  0,                                        // tp_getset
  0,                                        // tp_base
  0,                                        // tp_dict
  0,                                        // tp_descr_get
  0,                                        // tp_descr_set
  0,                                        // tp_dictoffset
  (initproc) dtype_init,                    // tp_init
  0,                                        // tp_alloc
  0                                         // tp_new
};

static bool dtype_has_valid_module() {
  dtypeType.tp_new = PyType_GenericNew;
  return PyType_Ready(&dtypeType) >= 0;
}

static void dtype_init_module(PyObject *module) {
  Py_INCREF(&dtypeType);
  PyModule_AddObject(module,
                     "dtype",
                     (PyObject*) &dtypeType);
}

OCCA_PY_MODULE(dtype, OCCA_PY_NO_METHODS)
//======================================

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
static int Kernel_init(occa::py::Kernel *self,
                       PyObject *args,
                       PyObject *kwargs) {
  self->kernel = NULL;

  occa::kernel kernel;
  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("kernel", kernel);

  if (!parser.parse(args, kwargs)) {
    return -1;
  }

  if (kernel.isInitialized()) {
    self->kernel = new occa::kernel(kernel);
  }

  return 0;
}

static void Kernel_dealloc(occa::py::Kernel *self) {
  delete self->kernel;
  Py_TYPE(self)->tp_free((PyObject*) self);
}
//======================================


//---[ Class Methods ]------------------
static PyObject* Kernel_is_initialized(occa::py::Kernel *self) {
  return occa::py::toPy(
    (bool) (self->kernel &&
            self->kernel->isInitialized())
  );
}

static PyObject* Kernel_free(occa::py::Kernel *self) {
  if (self->kernel) {
    self->kernel->free();
  }
  return occa::py::None();
}

static PyObject* Kernel_mode(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->mode()
  );
}

static PyObject* Kernel_properties(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->properties()
  );
}

static PyObject* Kernel_get_device(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->getDevice()
  );
}

static PyObject* Kernel_name(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->name()
  );
}

static PyObject* Kernel_source_filename(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->sourceFilename()
  );
}

static PyObject* Kernel_binary_filename(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->binaryFilename()
  );
}

static PyObject* Kernel_max_dims(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->maxDims()
  );
}

static PyObject* Kernel_max_outer_dims(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->maxOuterDims()
  );
}

static PyObject* Kernel_max_inner_dims(occa::py::Kernel *self) {
  if (!self->kernel) {
    return occa::py::None();
  }
  return occa::py::toPy(
    self->kernel->maxInnerDims()
  );
}

static PyObject* Kernel_set_run_dims(occa::py::Kernel *self,
                                     PyObject *args,
                                     PyObject *kwargs) {
  if (!self->kernel) {
    return occa::py::None();
  }

  occa::dim outer, inner;

  occa::py::kwargParser parser;
  parser
    .add("outer", outer)
    .add("inner", inner);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  self->kernel->setRunDims(outer, inner);

  return occa::py::None();
}

static bool occa_setKernelArg(PyObject *obj,
                              occa::kernelArg &arg) {
  // NULL or None
  if (occa::py::isNone(obj)) {
    arg = occa::kernelArg((void*) NULL);
    return true;
  }

  // occa::memory
  if (occa::py::isMemory(obj)) {
    arg = *(((occa::py::Memory*) obj)->memory);
    return true;
  }

  // numpy dtype
  if (occa::py::isNumpyScalar(obj)) {
    PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
    bool setArg = false;

#define CASE_TYPENUM(TYPENUM, CTYPE, SCALARTYPE)  \
    case TYPENUM:                                 \
      arg = (CTYPE) (((SCALARTYPE*) obj)->obval); \
      setArg = true;                              \
      break

    switch (descr->type_num) {
      CASE_TYPENUM(NPY_BOOL   , bool    , PyBoolScalarObject);
      CASE_TYPENUM(NPY_INT8   , int8_t  , PyInt8ScalarObject);
      CASE_TYPENUM(NPY_UINT8  , uint8_t , PyUInt8ScalarObject);
      CASE_TYPENUM(NPY_INT16  , int16_t , PyInt16ScalarObject);
      CASE_TYPENUM(NPY_UINT16 , uint16_t, PyUInt16ScalarObject);
      CASE_TYPENUM(NPY_INT32  , int32_t , PyInt32ScalarObject);
      CASE_TYPENUM(NPY_UINT32 , uint32_t, PyUInt32ScalarObject);
      CASE_TYPENUM(NPY_INT64  , int64_t , PyInt64ScalarObject);
      CASE_TYPENUM(NPY_UINT64 , uint64_t, PyUInt64ScalarObject);
      CASE_TYPENUM(NPY_FLOAT32, float   , PyFloat32ScalarObject);
      CASE_TYPENUM(NPY_FLOAT64, double  , PyFloat64ScalarObject);
    }
#undef CASE_TYPENUM

    Py_DECREF(descr);

    if (setArg) {
      return true;
    }
  }

  occa::py::raise("Unsupported type for a kernel argument");
  return false;
}

static PyObject* Kernel_run(occa::py::Kernel *self,
                            PyObject *args,
                            PyObject *kwargs) {
  if (!self->kernel) {
    return occa::py::None();
  }

  occa::py::list argList;

  occa::py::kwargParser parser;
  parser
    .add("args", argList);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  self->kernel->clearArgs();
  const int argc = argList.size();
  for (int i = 0; i < argc; ++i) {
    occa::kernelArg arg;
    if (!occa_setKernelArg(argList[i], arg)) {
      return NULL;
    }
    self->kernel->pushArg(arg);
  }

  self->kernel->run();

  return occa::py::None();
}
//======================================


//---[ Module ]-------------------------
#define KERNEL_METHOD_NO_ARGS(FUNC)             \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, Kernel_##FUNC)

#define KERNEL_METHOD_WITH_KWARGS(FUNC)             \
  OCCA_PY_METHOD_WITH_KWARGS(#FUNC, Kernel_##FUNC)

OCCA_PY_METHODS(
  Kernel_methods,
  KERNEL_METHOD_NO_ARGS(is_initialized),
  KERNEL_METHOD_NO_ARGS(free),
  KERNEL_METHOD_NO_ARGS(mode),
  KERNEL_METHOD_NO_ARGS(properties),
  KERNEL_METHOD_NO_ARGS(get_device),
  KERNEL_METHOD_NO_ARGS(name),
  KERNEL_METHOD_NO_ARGS(source_filename),
  KERNEL_METHOD_NO_ARGS(binary_filename),
  KERNEL_METHOD_NO_ARGS(max_dims),
  KERNEL_METHOD_NO_ARGS(max_outer_dims),
  KERNEL_METHOD_NO_ARGS(max_inner_dims),
  KERNEL_METHOD_WITH_KWARGS(set_run_dims),
  KERNEL_METHOD_WITH_KWARGS(run)
);

static PyTypeObject KernelType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "occa.c.Kernel",                          // tp_name
  sizeof(occa::py::Kernel),                 // tp_basicsize
  0,                                        // tp_itemsize
  (destructor) Kernel_dealloc,              // tp_dealloc
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
  "Wrapper for occa::kernel",               // tp_doc
  0,                                        // tp_traverse
  0,                                        // tp_clear
  0,                                        // tp_richcompare
  0,                                        // tp_weaklistoffset
  0,                                        // tp_iter
  0,                                        // tp_iternext
  Kernel_methods,                           // tp_methods
  0,                                        // tp_members
  0,                                        // tp_getset
  0,                                        // tp_base
  0,                                        // tp_dict
  0,                                        // tp_descr_get
  0,                                        // tp_descr_set
  0,                                        // tp_dictoffset
  (initproc) Kernel_init,                   // tp_init
  0,                                        // tp_alloc
  0                                         // tp_new
};

static bool kernel_has_valid_module() {
  KernelType.tp_new = PyType_GenericNew;
  return PyType_Ready(&KernelType) >= 0;
}

static void kernel_init_module(PyObject *module) {
  Py_INCREF(&KernelType);
  PyModule_AddObject(module,
                     "Kernel",
                     (PyObject*) &KernelType);
}

OCCA_PY_MODULE(kernel, OCCA_PY_NO_METHODS)
//======================================

#ifndef OCCA_PY_TOPY_HEADER
#define OCCA_PY_TOPY_HEADER

#include "defines.hpp"
#include "types.hpp"
#include "errors.hpp"


namespace occa {
  namespace py {
    typedef std::vector<npy_intp> npIntVector;

    // Special
    static PyObject* None() {
      Py_RETURN_NONE;
    }

    static PyObject* True() {
      Py_RETURN_TRUE;
    }

    static PyObject* False() {
      Py_RETURN_FALSE;
    }

    static PyObject* toPy(void *ptr) {
      if (ptr) {
        return PyCapsule_New(ptr, NULL, NULL);
      }
      return NULL;
    }

    // Bool
    static PyObject* toPy(const bool b) {
      if (b) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }

    // Integer Types
    static PyObject* toPy(const int value) {
      return PyLong_FromLong((long) value);
    }

    static PyObject* toPy(const long value) {
      return PyLong_FromLong(value);
    }

    static PyObject* toPy(const long long value) {
      return PyLong_FromLongLong(value);
    }

    static PyObject* toPy(const size_t value) {
      return PyLong_FromSize_t(value);
    }

    // Float / Double
    static PyObject* toPy(const float value) {
      return PyFloat_FromDouble((double) value);
    }

    static PyObject* toPy(const double value) {
      return PyFloat_FromDouble(value);
    }

    // String
    static PyObject* toPy(const char *c) {
      return PyUnicode_FromString(c);
    }

    static PyObject* toPy(const std::string s) {
      return PyUnicode_FromString(s.c_str());
    }

    // Dim
    static PyObject* toPy(const occa::dim d) {
      return Py_BuildValue("[iii]",
                           (int) d.x, (int) d.y, (int) d.z);
    }

    // Core
    static PyObject* newCoreType(PyTypeObject *Type,
                                 void *ptr,
                                 const std::string &name) {
      if (!ptr) {
        return occa::py::None();
      }

      PyObject *typeObj = occa::py::toPy(ptr);
      PyObject *typeArgs = Py_BuildValue("()");
      PyObject *typeKwargs = Py_BuildValue("{s:O}",
                                           name.c_str(), typeObj);

      PyObject *pyType = PyObject_Call((PyObject*) Type, typeArgs, typeKwargs);

      Py_XDECREF(typeObj);
      Py_XDECREF(typeArgs);
      Py_XDECREF(typeKwargs);

      return pyType;
    }

    static PyObject* toPy(const occa::device &device) {
      return newCoreType(occa::py::DeviceType(),
                         (void*) new occa::device(device),
                         "device");
    }

    static PyObject* toPy(const occa::kernel &kernel) {
      return newCoreType(occa::py::KernelType(),
                         (void*) new occa::kernel(kernel),
                         "kernel");
    }

    static PyObject* toPy(const occa::memory &memory) {
      return newCoreType(occa::py::MemoryType(),
                         (void*) new occa::memory(memory),
                         "memory");
    }

    static PyObject* toPy(const occa::stream &stream) {
      return newCoreType(occa::py::StreamType(),
                         (void*) new occa::stream(stream),
                         "stream");
    }

    static PyObject* toPy(const occa::streamTag &streamTag) {
      return newCoreType(occa::py::StreamTagType(),
                         (void*) new occa::streamTag(streamTag),
                         "streamtag");
    }

    // Props / JSON
    static PyObject* toPy(const occa::properties &props) {
      return toPy(props.dump(0));
    }

    static PyObject* toPy(const occa::json &j) {
      return toPy(j.dump(0));
    }

    // Numpy
    static PyObject* npArray(void *ptr,
                             npIntVector &dims,
                             int dtype_num) {

      return PyArray_SimpleNewFromData(
        (int) dims.size(),
        &(dims[0]),
        dtype_num,
        ptr
      );
    }
  }
}


#endif

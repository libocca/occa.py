#ifndef OCCA_PY_ERRORS_HEADER
#define OCCA_PY_ERRORS_HEADER

#include "defines.hpp"
#include "types.hpp"


#define OCCA_TRY_AND_RETURN(RETURN, ...)        \
  try {                                         \
    __VA_ARGS__                                 \
  } catch (occa::exception e) {                 \
    occa::py::raise(e);                         \
    return RETURN;                              \
  }

#define OCCA_TRY(...)                           \
  OCCA_TRY_AND_RETURN(NULL, __VA_ARGS__)


#define OCCA_INIT_TRY(...)                      \
  OCCA_TRY_AND_RETURN(-1, __VA_ARGS__)


namespace occa {
  namespace py {
    static void raise(occa::exception e) {
      std::string message = e.message;
      message += '\n';
      message += e.location();
      PyErr_SetString((PyObject*) ErrorType(),
                      message.c_str());
    }

    static void raise(const std::string &message) {
      PyErr_SetString((PyObject*) ErrorType(),
                      message.c_str());
    }
  }
}


#endif

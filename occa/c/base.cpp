#include "header.hpp"


//---[ Globals & Flags ]----------------
static PyObject* py_occa_settings(PyObject *self) {
  OCCA_TRY(
    return occa::py::toPy(
      occa::settings()
    );
  );
}

static PyObject* py_occa_set_setting(PyObject *self,
                                     PyObject *args,
                                     PyObject *kwargs) {
  std::string key;
  occa::json value;

  occa::py::kwargParser parser;
  parser
    .add("key", key)
    .add("value", value);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }
  OCCA_TRY(
    occa::settings()[key] = value;
  );
  return occa::py::None();
}
//======================================


//---[ Device ]-------------------------
static PyObject* py_occa_host(PyObject *self) {
  OCCA_TRY(
    return occa::py::toPy(
      occa::host()
    );
  );
}

static PyObject* py_occa_get_device(PyObject *self) {
  OCCA_TRY(
    return occa::py::toPy(
      occa::getDevice()
    );
  );
}

static PyObject* py_occa_set_device(PyObject *self,
                                    PyObject *args,
                                    PyObject *kwargs) {
  occa::device device;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .startOptionalKwargs()
    .add("device", device)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  OCCA_TRY(
    if (device.isInitialized()) {
      occa::setDevice(device);
    } else {
      occa::setDevice(props);
    }
  );

  return occa::py::None();
}
//======================================


//---[ Memory ]-------------------------
static PyObject* py_occa_memcpy(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs) {
  occa::py::memoryLike dest;
  occa::py::memoryLike src;
  long long bytes = -1;
  long long destOffset = 0;
  long long srcOffset = 0;
  occa::properties props;

  occa::py::kwargParser parser;
  parser
    .add("dest", dest)
    .add("src", src)
    .startOptionalKwargs()
    .add("bytes", bytes)
    .add("dest_offset", destOffset)
    .add("src_offset", srcOffset)
    .add("props", props);

  if (!parser.parse(args, kwargs)) {
    return NULL;
  }

  if (!dest.isInitialized()) {
    occa::py::raise("dest is not an occa.Memory or numpy.ndarray object");
    return NULL;
  }
  if (!src.isInitialized()) {
    occa::py::raise("src is not an occa.Memory or numpy.ndarray object");
    return NULL;
  }

  OCCA_TRY(
    if (dest.isMemory()) {
      if (src.isMemory()) {
        occa::memcpy(dest.memory(),
                     src.memory(),
                     bytes,
                     destOffset,
                     srcOffset,
                     props);
      } else {
        occa::memcpy(dest.memory(),
                     (void*) (src.ptr() + srcOffset),
                     bytes,
                     destOffset,
                     props);
      }
    } else {
      if (src.isMemory()) {
        occa::memcpy((void*) (dest.ptr() + destOffset),
                     src.memory(),
                     bytes,
                     srcOffset,
                     props);
      } else {
        occa::memcpy((void*) (dest.ptr() + destOffset),
                     (void*) (src.ptr() + srcOffset),
                     bytes,
                     props);
      }
    }
  );

  return occa::py::None();
}
//======================================
#define BASE_METHOD_NO_ARGS(FUNC)               \
  OCCA_PY_METHOD_NO_ARGS(#FUNC, py_occa_##FUNC)

#define BASE_METHOD_WITH_KWARGS(FUNC)               \
  OCCA_PY_METHOD_WITH_KWARGS(#FUNC, py_occa_##FUNC)


static bool base_has_valid_module() {
  return true;
}

static void base_init_module(PyObject *module) {}

OCCA_PY_MODULE(
  base,
  BASE_METHOD_NO_ARGS(settings),
  BASE_METHOD_WITH_KWARGS(set_setting),
  BASE_METHOD_NO_ARGS(host),
  BASE_METHOD_NO_ARGS(get_device),
  BASE_METHOD_WITH_KWARGS(set_device),
  BASE_METHOD_WITH_KWARGS(memcpy)
);

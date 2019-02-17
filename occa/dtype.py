import json
import numpy as np

from . import c, utils


__DTYPE_CACHE = None


def get_occa_dtype(dtype):
    global __DTYPE_CACHE

    if not __DTYPE_CACHE:
        __DTYPE_CACHE = get_dtype_cache()

    # Make sure we have a numpy dtype
    dtype = np.dtype(dtype)
    occa_dtype = __DTYPE_CACHE.get(dtype)

    if occa_dtype is None:
        occa_dtype = c.dtype(json=dtype_to_json(dtype))
        __DTYPE_CACHE[dtype] = occa_dtype

    return occa_dtype


def dtype_to_json(dtype, **kwargs):
    return json.dumps(dtype_to_obj(dtype), **kwargs)


def dtype_to_obj(dtype):
    # dtype tuple (np.float32, (2,2))
    if dtype.shape:
        count = 1
        for n in dtype.shape:
            count *= n
        [subdtype, *_] = dtype.subdtype
        return [dtype_to_obj(subdtype), count]

    # dtype(np.float32)
    if not dtype.fields:
        return utils.TYPES_TO_C_TYPES[dtype.type]

    # dtype([...])
    return [
        [field, dtype_to_obj(field_dtype)]
        for field, [field_dtype, *_] in dtype.fields.items()
    ]


def get_dtype_cache():
    return {
        np.dtype(np.bool_): c.dtype(builtin='bool'),
        np.dtype(np.int8): c.dtype(builtin='int8'),
        np.dtype(np.uint8): c.dtype(builtin='uint8'),
        np.dtype(np.int16): c.dtype(builtin='int16'),
        np.dtype(np.uint16): c.dtype(builtin='uint16'),
        np.dtype(np.int32): c.dtype(builtin='int32'),
        np.dtype(np.uint32): c.dtype(builtin='uint32'),
        np.dtype(np.int64): c.dtype(builtin='int64'),
        np.dtype(np.uint64): c.dtype(builtin='uint64'),
        np.dtype(np.float32): c.dtype(builtin='float'),
        np.dtype(np.float64): c.dtype(builtin='double'),
    }

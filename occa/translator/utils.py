import ast
import inspect
import sys

import numpy as np

from functools import reduce

VALID_PY_TYPES = {float, int, bool}

VALID_NP_TYPES = {
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
}


def join(sep, *args) -> str:
    """
        Will join a bunch of generators into a string
    """
    return sep.join(
        reduce(lambda x, y: x + list(y), args, [])
    )


def to_camel_case(function_name: str) -> str:
    """
        Function to turn a typical python naming
        convention into a OCCA-like camelCase
        function name

        add_vector -> addVectors
    """
    tokens = function_name.split("_")
    first = map(lambda token: token.lower(), tokens[:1])
    last = map(lambda token: token.title(), tokens[1:])
    return join("", first, last)


def make_ast(function):
    func_source = inspect.getsource(function)
    func_ast = ast.parse(func_source)
    return func_ast

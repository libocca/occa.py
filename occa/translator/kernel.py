import ast
import inspect
from collections import OrderedDict
from functools import wraps

import numpy as np

from .utils import VALID_PY_TYPES, VALID_NP_TYPES, make_ast, to_camel_case

ALLOWED_TYPES = VALID_PY_TYPES.union(VALID_NP_TYPES)


def make_c_type(dtype):
    _type = np.dtype(dtype)
    return _type.name + "_t" if _type != np.dtype(bool) else _type.name


def make_signiture(arg):
    """
        Arg is a key, value pair of argument name
        and the type given from __annotations__

        This only supports python built-in int and float primiatives
    """
    _var_name, _type = arg
    if isinstance(_type, ast.List):
        _type = _type.elts[0]
        _var_name = "*" + _var_name.replace("*", "")
        return make_signiture((_var_name, _type))
    elif isinstance(_type, ast.Attribute):
        if _type.value.id == 'np':
            _type = make_c_type(_type.attr)
        else:
            msg = "Kernal translator only supports Python or numpy arithmetic types"
            raise RuntimeError(msg)
    elif isinstance(_type, ast.Str) or _type.id == 'str':
        msg = "varible {} is of type str and not supported".format(_var_name)
        raise RuntimeError(msg)
    elif isinstance(_type, ast.Name):
        _type = make_c_type(_type.id)
    else:
        raise RuntimeError("Do I ever get here?")
    return "{} {}".format(_type, _var_name)


def kernel(function):
    """
        This is the core logic of converting a
        python AST to the OCCA language


    """
    if not inspect.isfunction(function):
        msg = """Can only convert functions into OCCA Kernels"""
        raise RuntimeError(msg)

    @wraps(function)
    def wrapper(*args, **kwargs):
        func_ast = make_ast(function)
        func_name = func_ast.body[0].name
        occa_function_name = to_camel_case(func_name)

        occa_source = "@kernel void {}(".format(occa_function_name)

        arguments = OrderedDict()
        for arg in func_ast.body[0].args.args:
            if not arg.annotation:
                msg = "Argument {} in function {} has no type def".format(arg.arg,
                                                                          func_name)
                raise RuntimeError(msg)
            arguments[arg.arg] = arg.annotation

        occa_source += ', '.join(map(make_signiture, arguments.items()))
        occa_source += "){\n}"
        print(occa_source)

    return wrapper

#
# The MIT License (MIT)
#
# Copyright (c) 2018 David Medina
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#
import dis
import opcode
import types
import numpy as np


pytype_to_ctype = {
    np.bool_: 'bool',
    np.int8: 'int8_t',
    np.int16: 'int16_t',
    np.int32: 'int32_t',
    np.int64: 'int64_t',
    np.uint8: 'uint8_t',
    np.uint16: 'uint16_t',
    np.uint32: 'uint32_t',
    np.uint64: 'uint64_t',
    np.float32: 'float',
    np.float64: 'double',
}


def get_ctype(pytype):
    origin = getattr(pytype, '__origin__', None)
    if ((origin is not None and origin != List) or
        (origin is None and pytype not in pytype_to_ctype)):
        raise ValueError('Unable to handle type: [{}]'.format(pytype))

    if origin:
        base_type = get_ctype(pytype.__args__[0])
        return '{base_type}*'.format(base_type=base_type)
    return '{type} '.format(type=pytype_to_ctype[pytype])


def make_kernel(func):
    c = func.__code__
    name = c.co_name
    args = c.co_varnames[:c.co_argcount]
    arg_types = typing.get_type_hints(func)
    # Make sure typings are set
    if function_has_return_value(func):
        raise ValueError("Kernels shouldn't return values")
    missing_args = [
        arg
        for arg in args
        if arg not in arg_types
    ]
    if len(missing_args):
        raise ValueError('Function is missing argument types')
    # Get argument types
    c_args = ', '.join((
        '{c_type}{arg}'.format(c_type=get_ctype(arg_types[arg]),
                               arg=arg)
        for arg in args
    ))
    print('''
@kernel void {name}({args}) {{
}}
'''.format(name=name,
           args=c_args))


def bytecode_function(name, bytecode, **kwargs):
    def _(): pass
    args = {
        'co_argcount': 0,
        'co_cellvars': (),
        'co_code': bytecode,
        'co_consts': (None,),
        'co_filename': '<source>',
        'co_firstlineno': 1,
        'co_flags': 67,
        'co_freevars': (),
        'co_kwonlyargcount': 0,
        'co_lnotab': b'',
        'co_name': name,
        'co_names': (),
        'co_nlocals': 0,
        'co_stacksize': 0,
        'co_varnames': (),
    }
    args.update(kwargs)

    _.__code__ = types.CodeType(
        args['co_argcount'],
        args['co_kwonlyargcount'],
        args['co_nlocals'],
        args['co_stacksize'],
        args['co_flags'],
        args['co_code'],
        args['co_consts'],
        args['co_names'],
        args['co_varnames'],
        args['co_filename'],
        args['co_name'],
        args['co_firstlineno'],
        args['co_lnotab'],
        args['co_freevars'],
        args['co_cellvars'],
    )
    return _

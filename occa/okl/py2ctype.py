import ast

from . import utils
from . import oklifier as oklifier_module
from .exceptions import TransformError


PY_TO_C_TYPES = {
    'void': 'void',
    'bool': 'bool',
    'int': 'int',
    'float': 'double',
    'str': 'char *',
    'bool_': 'bool',
    'int8': 'char',
    'uint8': 'char',
    'int16': 'short',
    'uint16': 'short',
    'int32': 'int',
    'uint32': 'int',
    'int64': 'long',
    'uint64': 'long',
    'float32': 'float',
    'float64': 'double',
    'np.bool_': 'bool',
    'np.int8': 'char',
    'np.uint8': 'char',
    'np.int16': 'short',
    'np.uint16': 'short',
    'np.int32': 'int',
    'np.uint32': 'int',
    'np.int64': 'long',
    'np.uint64': 'long',
    'np.float32': 'float',
    'np.float64': 'double',
}


class Py2CType:
    def __init__(self,
                 node,
                 varname='',
                 oklifier=None):
        self.root = node
        self.varname = varname

        self.has_oklifier_origin = oklifier is not None
        self.oklifier = oklifier or oklifier_module.Oklifier(node)

        # Set to true when we need to stop trying to map Py -> C
        # For example:
        #   Const[Array[np.float32]] -> const [[float] *]
        #                                       ^
        #                    Don't map after this
        self.found_c_type = False
        # Set to True of varname is empty
        self.applied_varname = not bool(varname)

    def to_c(self):
        if self.root is None:
            return None

        return self.node_to_c(self.root)

    def node_to_c(self, node):
        type_str = self.stringify_node(node)

        if not self.found_c_type:
            self.found_c_type = True
            c_type_str = PY_TO_C_TYPES.get(type_str)
            if c_type_str is None:
                self.raise_error(node,
                                 'Cannot convert Python type annotation [{}] to a C type'
                                 .format(type_str))
            type_str = c_type_str

        return self.add_varname(type_str)

    def stringify_node(self, node):
        if isinstance(node, str):
            return node

        node_type = type(node)
        if node_type is ast.Name:
            return self.oklifier.stringify_Name(node)

        elif node_type is ast.Num:
            return self.oklifier.stringify_Num(node)

        elif node_type is ast.NameConstant:
            node_str = self.oklifier.stringify_NameConstant(node)
            if node_str == 'NULL':
                return 'void'

        elif node_type is ast.Attribute:
            return self.oklifier.stringify_Attribute(node)

        elif node_type is ast.Subscript:
            value, index = self.oklifier.split_subscript(node)
            value_str = self.oklifier.stringify_node(value)

            if value_str == 'Array':
                return self.stringify_array(index)
            if value_str == 'Const':
                return 'const ' + self.node_to_c(index)
            if value_str == 'Exclusive':
                return '@exclusive ' + self.node_to_c(index)
            if value_str == 'Shared':
                return '@shared ' + self.node_to_c(index)

        self.raise_error(node,
                         'Cannot convert Python type annotation to a C type')

    def stringify_array(self, node):
        # Avoid adding varname while getting the inner type
        prev_applied_varname = self.applied_varname
        self.applied_varname = True

        if not isinstance(node, ast.Tuple):
            type_str = self.node_to_c(node)
            self.applied_varname = prev_applied_varname

            return self.add_varname(type_str, '*')

        type_node, *indices = node.elts
        type_str = self.node_to_c(type_node)
        arrays_str = ''.join((
            '[{index}]'.format(index=self.stringify_node(index))
            for index in indices
        ))

        self.applied_varname = prev_applied_varname
        return self.add_varname(type_str) + arrays_str

    @staticmethod
    def add_space(type_str):
        if (type_str.endswith('*') or
            type_str.endswith(' ')):
            return type_str
        return type_str + ' '

    def add_varname(self, type_str, between=''):
        if between:
            type_str = self.add_space(type_str) + between

        if self.applied_varname:
            return type_str

        if not between:
            type_str = self.add_space(type_str)

        self.applied_varname = True
        return type_str + self.varname

    def raise_error(self, node, message):
        if self.has_oklifier_origin:
            self.oklifier.raise_error(node, message)

        error_message = utils.get_node_error_message(node, message)
        raise TransformError(error_message)


def py2ctype(node, varname='', oklifier=None):
    return Py2CType(node, varname, oklifier).to_c()

import ast
import inspect


UnaryOp_formats = {
    ast.Invert: '~{value}',
    ast.Not: '!{value}',
    ast.UAdd: '+{value}',
    ast.USub: '-{value}',
}


BinOp_formats = {
    ast.Add: '{left} + {right}',
    ast.Sub: '{left} - {right}',
    ast.Mult: '{left} * {right}',
    ast.Div: '{left} / {right}',
    ast.Mod: '{left} % {right}',
    ast.Pow: 'pow({left}, {right})',
    ast.LShift: '{left} << {right}',
    ast.RShift: '{left} >> {right}',
    ast.BitOr: '{left} | {right}',
    ast.BitAnd: '{left} & {right}',
    ast.FloorDiv: 'floor({left} / {right})',
}


CompareOp_str = {
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Is: '==',
    ast.IsNot: '!=',
}


OklAttr_str = {
    '@okl.kernel': '@kernel',
}


class Oklifier:
    def __init__(self, obj):
        self.obj = obj
        self.source = inspect.getsource(obj)
        self.root = ast.parse(self.source)

        self.stringify_node_map = {
            ast.Assign: self.Assign_str,
            ast.Attribute: self.Attribute_str,
            ast.BinOp: self.BinOp_str,
            ast.BoolOp: self.BoolOp_str,
            ast.Compare: self.Compare_str,
            ast.Index: self.Index_str,
            ast.Expr: self.Expr_str,
            ast.For: self.For_str,
            ast.FunctionDef: self.FunctionDef_str,
            ast.Module: self.Module_str,
            ast.Name: self.Name_str,
            ast.NameConstant: self.NameConstant_str,
            ast.Num: self.Num_str,
            ast.Subscript: self.Subscript_str,
            ast.UnaryOp: self.UnaryOp_str,
        }

    def Module_str(self, node, indent=''):
        return self.stringify_nodes(node.body, indent)

    def Name_str(self, node, indent=''):
        return node.id

    def NameConstant_str(self, node, indent=''):
        value = node.value
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        if value is None:
            return 'void'
        self.__raise_error(node,
                           'Cannot handle NameConstant')

    def split_subscript(self, node):
        if not isinstance(node.slice, ast.Index):
            self.__raise_error(node.slice,
                               'Can only handle single access slices')
        return [node.value, node.slice]

    def Subscript_str(self, node, indent=''):
        value, index = self.split_subscript(node)
        return '{value}[{index}]'.format(value=self.stringify_node(value, indent),
                                         index=self.stringify_node(index, indent))

    def Index_str(self, node, indent=''):
        return self.stringify_node(node.value, indent)

    def Assign_str(self, node, indent=''):
        if len(node.targets) != 1:
            self.__raise_error(node,
                               'Cannot handle assignment of more than 1 value')
        return '{left} = {right}'.format(left=self.stringify_node(node.targets[0], indent),
                                         right=self.stringify_node(node.value, indent))

    def Expr_str(self, node, indent=''):
        return self.stringify_node(node.value, indent)

    def Num_str(self, node, indent=''):
        return node.n


    def UnaryOp_str(self, node, indent=''):
        str_format = UnaryOp_formats.get(type(node.op))
        if str_format is None:
            self.__raise_error(node.op,
                               'Unable to handle operator')
        return str_format.format(value=self.stringify_node(node.operand, indent))

    def BinOp_str(self, node, indent=''):
        str_format = BinOp_formats.get(type(node.op))
        if str_format is None:
            self.__raise_error(node.op,
                               'Unable to handle operator')
        return str_format.format(left=self.stringify_node(node.left, indent),
                                 right=self.stringify_node(node.right, indent))

    def BoolOp_str(self, node, indent=''):
        op = ' && ' if isinstance(node.op, ast.And) else ' || '
        # TODO: Parentheses only if needed
        return '({values})'.format(values=op.join(
            self.stringify_node(subnode, indent)
            for subnode in node.values
        ))

    def Compare_str(self, node, indent=''):
        ops = [type(op) for op in node.ops]
        for op_index, op in enumerate(ops):
            if op not in CompareOp_str:
                self.__raise_error(node.ops[op_index],
                                   'Cannot handle comparison operator')
            ops = [
                CompareOp_str[op]
            for op in ops
            ]
            values = [
                self.stringify_node(subnode, indent)
            for subnode in [node.left, *node.comparators]
            ]
            # TODO: Parentheses only if needed
        return ' && '.join(
            '({left} {op} {right})'.format(left=left,
                                           right=right,
                                           op=ops[index])
            for index, (left, right) in enumerate(zip(values[:-1], values[1:]))
        )

    def split_for_iter(self, node):
        # TODO: Extract start, end, step from node
        return 0, 10, 1

    def For_str(self, node, indent=''):
        if not isinstance(node.target, ast.Name):
            self.__raise_error(node.target,
                               'Can only handle one variable for the for-loop index')
        index = self.stringify_node(node.target, indent)
        start, end, step = self.split_for_iter(node.iter)

        if step == 0:
            self.__raise_error(node.iter,
                               'Cannot have for-loop with a step size of 0')

        if step > 0:
            if step == 1:
                step = '++{index}'.format(index=index)
            else:
                step = '{index} += {step}'.format(index=index, step=step)
        else:
            if step == -1:
                step = '--{index}'.format(index=index)
            else:
                step = '{index} -= {step}'.format(index=index, step=-step)

        for_str = (
            'for (int {index} = {start}; {index} < {end}; {step}) {{'
        ).format(index=index,
                 start=start,
                 end=end,
                 step=step)

        body = self.stringify_nodes(node.body, indent + '  ')
        if body:
            for_str += '\n{body}\n{indent}'.format(body=body,
                                                   indent=indent)
        return for_str + '}'


    def Attribute_str(self, node, indent=''):
        attr = '@{name}.{value}'.format(name=self.stringify_node(node.value, indent),
                                        value=node.attr)
        okl_attr = OklAttr_str.get(attr)
        if okl_attr is None:
            self.__raise_error(node,
                               'Cannot handle attribute')
        return okl_attr

    def TypeAnnotation_str(self, node):
        if node is None:
            return None

        node_type = type(node)
        if node_type is ast.Name:
            return self.py_to_c_type(self.Name_str(node))
        if node_type is ast.NameConstant:
            return self.NameConstant_str(node)
        if node_type is ast.Index:
            return self.Index_str(node)
        if node_type is ast.Subscript or node_type is ast.List:
            found_error = False
            if node_type is ast.Subscript:
                value, index = self.split_subscript(node)
                if self.stringify_node(value) != 'List':
                    found_error = True
            if node_type is ast.List:
                indices = node.elts
                if len(indices) != 1:
                    self.__raise_error(node,
                                       'Can only handle single-valued lists as arguments')
                index = indices[0]
            # Make sure no errors were found
            if not found_error:
                type_str = self.TypeAnnotation_str(index)
                if not type_str.endswith('*'):
                    type_str += ' '
                return type_str + '*'


        self.__raise_error(node,
                           'Cannot handle type annotation')

    def Arguments_str(self, args, indent=''):
        if args.kwarg:
            self.__raise_error(args, 'Cannot handle **kwargs')
        if args.vararg:
            self.__raise_error(args, 'Cannot handle *args')
        if args.defaults or args.kw_defaults:
            self.__raise_error(args, 'Cannot handle default arguments yet')

        args = [
            *args.args,
            *args.kwonlyargs,
        ]
        arg_count = len(args)

        args_str = ''
        for index, arg in enumerate(args):
            type_annotation = self.TypeAnnotation_str(arg.annotation)
            if type_annotation is None:
                self.__raise_error(arg,
                                   'Arguments must have a type annotation')

            if not type_annotation.endswith('*'):
                type_annotation += ' '

            args_str += '{type}{arg}'.format(type=type_annotation,
                                             arg=arg.arg)
            if index < (arg_count - 1):
                args_str += ',\n' + indent

        return args_str

    def FunctionDef_str(self, node, indent=''):
        name = node.name

        if node.returns is None:
            self.__raise_error(node,
                               'Function must have a return value type')

        returns = self.TypeAnnotation_str(node.returns)

        # TODO: Make sure they are only supported OKL attributes like @kernel
        decorators = ' '.join(
            self.stringify_node(decorator)
            for decorator in node.decorator_list
        )
        if decorators:
            decorators += ' '

        func_str = (
            '{decorators}{returns} {name}('
        ).format(decorators=decorators,
                 name=name,
                 returns=returns)

        args = self.Arguments_str(node.args, indent=(' ' * len(func_str)))

        func_str += '{args}) {{'.format(args=args)

        body = self.stringify_nodes(node.body, indent + '  ')
        if body:
            func_str += '\n{body}\n{indent}'.format(body=body,
                                                    indent=indent)
        return func_str + '}'

    def stringify_node(self, node, indent=''):
        node_str_func = self.stringify_node_map.get(type(node))
        if node_str_func is not None:
            return node_str_func(node, indent)
        self.self.__raise_error(node, 'Unable to handle node type')

    def stringify_nodes(self, nodes, indent=''):
        return '\n'.join(
            (indent + self.stringify_node(node, indent=indent))
            for node in nodes
        )

    def py_to_c_type(self, type_name):
        return type_name

    def __raise_error(self, node, message):
        error_line = node.lineno - 1
        char_pos = node.col_offset

        error_message = 'Error: {message}\n'.format(message=message)

        # Get context lines
        lines = [
            line
            for line in range(error_line - 2, error_line + 3)
            if line >= 0
        ]
        # Stringify lines and pad them
        lines_str = [str(line) for line in lines]
        char_size = max(len(line) for line in lines_str)
        lines_str = [line.ljust(char_size) for line in lines_str]

        source_lines = self.source.splitlines()

        for index, line in enumerate(lines):
            error_message += lines_str[index] + ' | ' + source_lines[line] + '\n'
            if line == error_line:
                error_message += ' ' * char_size
                error_message += ' | '
                error_message += ' ' * char_pos
                error_message += '^\n'

        raise ValueError(error_message)

    def to_str(self):
        return self.stringify_node(self.root)


def py2okl(func):
    return Oklifier(func).to_str()

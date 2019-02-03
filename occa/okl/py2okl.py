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


OklDecorators_str = {
    '@okl.kernel': '@kernel',
}

foobar = None
class Oklifier:
    def __init__(self, obj):
        self.obj = obj
        self.source = None
        if isinstance(obj, ast.AST):
            self.root = obj
        else:
            self.source = inspect.getsource(obj)
            self.root = ast.parse(self.source)

        self.stringify_node_map = {
            ast.Assign: self.stringify_Assign,
            ast.Attribute: self.stringify_Attribute,
            ast.BinOp: self.stringify_BinOp,
            ast.BoolOp: self.stringify_BoolOp,
            ast.Call: self.stringify_Call,
            ast.Compare: self.stringify_Compare,
            ast.Index: self.stringify_Index,
            ast.Expr: self.stringify_Expr,
            ast.For: self.stringify_For,
            ast.FunctionDef: self.stringify_FunctionDef,
            ast.Module: self.stringify_Module,
            ast.Name: self.stringify_Name,
            ast.NameConstant: self.stringify_NameConstant,
            ast.Num: self.stringify_Num,
            ast.Subscript: self.stringify_Subscript,
            ast.UnaryOp: self.stringify_UnaryOp,
        }

    def stringify_Arguments(self, args, indent=''):
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
            type_annotation = self.stringify_TypeAnnotation(arg.annotation)
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

    def stringify_Assign(self, node, indent=''):
        if len(node.targets) != 1:
            self.__raise_error(node,
                               'Cannot handle assignment of more than 1 value')
        return '{left} = {right}'.format(left=self.stringify_node(node.targets[0], indent),
                                         right=self.stringify_node(node.value, indent))

    def stringify_Attribute(self, node, indent=''):
        return '{name}.{value}'.format(name=self.stringify_node(node.value, indent),
                                       value=node.attr)

    def stringify_BinOp(self, node, indent=''):
        str_format = BinOp_formats.get(type(node.op))
        if str_format is None:
            self.__raise_error(node.op,
                               'Unable to handle operator')
        return str_format.format(left=self.stringify_node(node.left, indent),
                                 right=self.stringify_node(node.right, indent))

    def stringify_BoolOp(self, node, indent=''):
        op = ' && ' if isinstance(node.op, ast.And) else ' || '
        # TODO: Parentheses only if needed
        return '({values})'.format(values=op.join(
            self.stringify_node(subnode, indent)
            for subnode in node.values
        ))

    def stringify_Call(self, node, indent=''):
        args = ', '.join(
            self.stringify_node(arg)
            for arg in node.args
        )
        return '{indent}{func}({args})'.format(
            indent=indent,
            func=self.stringify_node(node.func),
            args=args,
        )

    def stringify_Compare(self, node, indent=''):
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

    def stringify_Index(self, node, indent=''):
        return self.stringify_node(node.value, indent)

    def stringify_Expr(self, node, indent=''):
        return self.stringify_node(node.value, indent)

    def split_for_iter(self, node):
        self.stringify_node(node)
        # TODO: Extract start, end, step from node
        return 0, 10, 1

    def stringify_For(self, node, indent=''):
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

    def stringify_FunctionDef(self, node, indent=''):
        name = node.name

        if node.returns is None:
            self.__raise_error(node,
                               'Function must have a return value type')

        returns = self.stringify_TypeAnnotation(node.returns)

        # TODO: Make sure they are only supported OKL attributes like @kernel
        decorators = ''
        for decorator in node.decorator_list:
            decorator_str = '@{attr}'.format(attr=self.stringify_node(decorator))
            okl_decorator = OklDecorators_str.get(decorator_str)
            if okl_decorator is None:
                self.__raise_error(decorator,
                                   'Cannot handle decorator')
            decorators += okl_decorator + ' '

        func_str = (
            '{decorators}{returns} {name}('
        ).format(decorators=decorators,
                 name=name,
                 returns=returns)

        args = self.stringify_Arguments(node.args, indent=(' ' * len(func_str)))

        func_str += '{args}) {{'.format(args=args)

        body = self.stringify_nodes(node.body, indent + '  ')
        if body:
            func_str += '\n{body}\n{indent}'.format(body=body,
                                                    indent=indent)
        return func_str + '}'

    def stringify_Module(self, node, indent=''):
        return self.stringify_nodes(node.body, indent)

    def stringify_Name(self, node, indent=''):
        return node.id

    def stringify_NameConstant(self, node, indent=''):
        value = node.value
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        if value is None:
            return 'void'
        self.__raise_error(node,
                           'Cannot handle NameConstant')

    def stringify_Num(self, node, indent=''):
        return str(node.n)

    def split_subscript(self, node):
        if not isinstance(node.slice, ast.Index):
            self.__raise_error(node.slice,
                               'Can only handle single access slices')
        return [node.value, node.slice]

    def stringify_Subscript(self, node, indent=''):
        value, index = self.split_subscript(node)
        return '{value}[{index}]'.format(value=self.stringify_node(value, indent),
                                         index=self.stringify_node(index, indent))

    def stringify_TypeAnnotation(self, node):
        if node is None:
            return None

        node_type = type(node)
        if node_type is ast.Name:
            return self.py_to_c_type(self.stringify_Name(node))
        if node_type is ast.NameConstant:
            return self.stringify_NameConstant(node)
        if node_type is ast.Index:
            return self.stringify_Index(node)
        if node_type is ast.Subscript:
            value, index = self.split_subscript(node)
            if self.stringify_node(value) == 'List':
                type_str = self.stringify_TypeAnnotation(index)
                if not type_str.endswith('*'):
                    type_str += ' '
                return type_str + '*'

        self.__raise_error(node,
                           'Cannot handle type annotation')

    def stringify_UnaryOp(self, node, indent=''):
        str_format = UnaryOp_formats.get(type(node.op))
        if str_format is None:
            self.__raise_error(node.op,
                               'Unable to handle operator')
        return str_format.format(value=self.stringify_node(node.operand, indent))

    def stringify_node(self, node, indent=''):
        node_str_func = self.stringify_node_map.get(type(node))
        if node_str_func is not None:
            return node_str_func(node, indent)
        self.__raise_error(node, 'Unable to handle node type {}'.format(type(node)))

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

        if self.source:
            source_lines = self.source.splitlines()

            # Get context lines
            lines = [
                line
                for line in range(error_line - 2, error_line + 3)
                if 0 <= line < len(source_lines)
            ]
            # Stringify lines and pad them
            lines_str = [str(line + 1) for line in lines]
            char_size = max(len(line) for line in lines_str)
            lines_str = [line.ljust(char_size) for line in lines_str]

            prefix = '   '

            for index, line in enumerate(lines):
                error_message += prefix
                error_message += lines_str[index]
                error_message += ' | '
                error_message += source_lines[line] + '\n'
                if line == error_line:
                    error_message += prefix
                    error_message += ' ' * char_size
                    error_message += ' | '
                    error_message += (' ' * char_pos) + '^\n'

        raise ValueError(error_message)

    def to_str(self):
        return self.stringify_node(self.root)


def py2okl(func):
    return Oklifier(func).to_str()

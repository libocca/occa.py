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


def NodeBody_str(nodes, indent):
    return '\n'.join(
        (indent + node_str(node, indent=indent))
        for node in nodes
    )


def Module_str(node, indent):
    return NodeBody_str(node.body, indent)


def Name_str(node, indent):
    return node.id


def Subscript_str(node, indent):
    if not isinstance(node.slice, ast.Index):
        raise ValueError('Can only handle single access slices')
    return '{value}[{index}]'.format(value=node_str(node.value, indent),
                                     index=node_str(node.slice, indent))


def Index_str(node, indent):
    return node_str(node.value, indent)


def Assign_str(node, indent):
    if len(node.targets) != 1:
        raise ValueError('Cannot handle assignment of more than 1 value')
    return '{left} = {right}'.format(left=node_str(node.targets[0], indent),
                                     right=node_str(node.value, indent))


def Expr_str(node, indent):
    return node_str(node.value, indent)


def Num_str(node, indent):
    return node.n



def UnaryOp_str(node, indent):
    str_format = UnaryOp_formats.get(type(node.op))
    if str_format is None:
        raise ValueError('Unable to handle operator: {}'.format(type(node.op)))
    return str_format.format(value=node_str(node.operand, indent))



def BinOp_str(node, indent):
    str_format = BinOp_formats.get(type(node.op))
    if str_format is None:
        raise ValueError('Unable to handle operator: {}'.format(type(node.op)))
    return str_format.format(left=node_str(node.left, indent),
                             right=node_str(node.right, indent))


def BoolOp_str(node, indent):
    op = ' && ' if isinstance(node.op, ast.And) else ' || '
    # TODO: Parentheses only if needed
    return '({values})'.format(values=op.join(
        node_str(subnode, indent)
        for subnode in node.values
    ))



def Compare_str(node, indent):
    ops = [type(op) for op in node.ops]
    for op in ops:
        if op not in CompareOp_str:
            raise ValueError('Cannot handle comparison operator: {}'.format(op))
        ops = [
            CompareOp_str[op]
        for op in ops
        ]
        values = [
            node_str(subnode, indent)
        for subnode in [node.left, *node.comparators]
        ]
        # TODO: Parentheses only if needed
    return ' && '.join(
        '({left} {op} {right})'.format(left=left,
                                       right=right,
                                       op=ops[index])
        for index, (left, right) in enumerate(zip(values[:-1], values[1:]))
    )


def For_split_iter(node):
    # TODO: Extract start, end, step from node
    return 0, 10, 1

def For_str(node, indent):
    if not isinstance(node.target, ast.Name):
        raise ValueError('Can only handle one variable for the for-loop index')
    index = node_str(node.target, indent)
    start, end, step = For_split_iter(node.iter)

    if step == 0:
        raise ValueError('Cannot have for-loop with a step size of 0')

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

    body = NodeBody_str(node.body, indent + '  ')
    if body:
        for_str += '\n{body}\n{indent}'.format(body=body,
                                               indent=indent)
    return for_str + '}'


OKLAttr_str = {
    '@okl.kernel': '@kernel',
}

def Attribute_str(node, indent):
    attr = '@{name}.{value}'.format(name=node_str(node.value, indent),
                                    value=node.attr)
    okl_attr = OKLAttr_str.get(attr)
    if okl_attr is None:
        raise ValueError('Cannot handle attribute: {attr}'.format(attr=attr))
    return okl_attr


def FunctionDef_str(node, indent):
    name = 'add_vectors'
    args = '...'
    returns = 'void'

    # TODO: Make sure they are only supported OKL attributes like @kernel
    decorators = ' '.join(
        node_str(decorator)
        for decorator in node.decorator_list
    )
    if decorators:
        decorators += ' '

    func_str = (
        '{decorators}{returns} {name}({args}) {{'
    ).format(decorators=decorators,
             name=name,
             args=args,
             returns=returns)

    body = NodeBody_str(node.body, indent + '  ')
    if body:
        func_str += '\n{body}\n{indent}'.format(body=body,
                                                indent=indent)
    return func_str + '}'


Node_str = {
    ast.Assign: Assign_str,
    ast.Attribute: Attribute_str,
    ast.BinOp: BinOp_str,
    ast.BoolOp: BoolOp_str,
    ast.Compare: Compare_str,
    ast.Index: Index_str,
    ast.Expr: Expr_str,
    ast.For: For_str,
    ast.FunctionDef: FunctionDef_str,
    ast.Module: Module_str,
    ast.Name: Name_str,
    ast.Num: Num_str,
    ast.Subscript: Subscript_str,
    ast.UnaryOp: UnaryOp_str,
}


def node_str(node, indent=''):
    str_func = Node_str.get(type(node))
    if str_func is None:
        raise ValueError('Unable to handle node type: {}'.format(type(node)))
    return str_func(node, indent)


def py2okl(func):
    root = ast.parse(inspect.getsource(func))
    return node_str(root)
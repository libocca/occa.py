import ast
import collections
import numpy as np

from . import attrorderer, oklifier
from .exceptions import TransformError


def flatten(obj):
    if isinstance(obj, collections.Iterable):
        return [
            item
            for obj_item in obj
            for item in flatten(obj_item)
        ]
    return [obj]


def get_attribute_chain(node):
    return attrorderer.AttrOrderer(node)


def get_node_error_message(node,
                           message='',
                           source=None,
                           source_indent_size=0):
    error_line = node.lineno - 1
    char_pos = node.col_offset + source_indent_size

    if message:
        error_message = 'Error: {message}\n'.format(message=message)
    else:
        error_message = 'Error:\n'.format(message=message)

    if not source:
        return error_message

    source_lines = source.splitlines()

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

    return error_message


def py2okl(obj, *, globals=None, _show_full_stacktrace=False):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, collections.Iterable):
        return [
            py2okl(item,
                   globals=globals,
                   _show_full_stacktrace=_show_full_stacktrace)
            for item in obj
        ]
    if not _show_full_stacktrace:
        try:
            return oklifier.Oklifier(obj, globals=globals).to_str()
        except TransformError as e:
            raise TransformError(str(e)) from None
    else:
        return oklifier.Oklifier(obj, globals=globals).to_str()

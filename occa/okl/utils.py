import ast
import collections

from . import attrorderer, oklifier


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


def py2okl(obj, globals=None):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, collections.Iterable):
        return [
            py2okl(item, globals=globals)
            for item in obj
        ]
    return oklifier.Oklifier(obj, globals=globals).to_str()

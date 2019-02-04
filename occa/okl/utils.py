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


def py2okl(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, collections.Iterable):
        return [
            py2okl(item)
            for item in obj
        ]
    return oklifier.Oklifier(obj).to_str()

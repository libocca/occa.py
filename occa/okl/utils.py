import collections

from . import attrorderer, oklifier


def get_attribute_chain(node):
    return attrorderer.AttrOrderer(node)


def py2okl(obj):
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, collections.Iterable):
        return oklifier.Oklifier(obj).to_str()
    return [
        oklifier.Oklifier(item).to_str()
        for item in obj
    ]

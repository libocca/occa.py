import collections

from .oklifier import Oklifier


def py2okl(obj):
    if not isinstance(obj, collections.Iterable):
        return Oklifier(obj).to_str()
    return [
        Oklifier(item).to_str()
        for item in obj
    ]

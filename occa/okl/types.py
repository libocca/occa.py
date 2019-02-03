from typing import Any


class StaticGetItem(type):
    def __getitem__(cls, other_type):
        pass

class Const(object, metaclass=StaticGetItem):
    pass


def dtype(type, *args):
    return {'type': type, 'qualifiers': args}


def shared(dtype, size):
    pass


def exclusive(dtype):
    pass

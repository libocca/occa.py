from typing import Any


class StaticGetItem(type):
    def __getitem__(cls, other_type):
        pass


class Array(object, metaclass=StaticGetItem):
    pass


class Const(object, metaclass=StaticGetItem):
    pass


class Exclusive(object, metaclass=StaticGetItem):
    pass


class Shared(object, metaclass=StaticGetItem):
    pass

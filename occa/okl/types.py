const = 'const'
restrict = '@restrict'


def dtype(type, *args):
    return {'type': type, 'qualifiers': args}


def shared(size, dtype=None):
    pass


def exclusive(dtype=None):
    pass

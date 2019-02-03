const = 'const'
restrict = '@restrict'


def dtype(type, *args):
    return {'type': type, 'qualifiers': args}


def shared(dtype, size):
    pass


def exclusive(dtype):
    pass

class Range:
    def __init__(self, start, stop=None, step=None):
        self.start = start
        self.stop = stop
        self.step = step

        if stop is None:
            self.stop = start
            self.start = 0

        if step is None:
            self.step = 1

    def to_json(self, **kwargs):
        return kwargs

    @property
    def outer(self):
        pass

    @property
    def inner(self):
        pass

    def tile(self, *args):
        pass


def range(start, stop=None, step=None):
    return Range(start, stop, step)

class Range:
    def __init__(self, start, stop=None, step=None):
        self.start = start
        self.stop = stop
        self.step = step

        self._is_outer = False
        self._is_inner = False
        self._tiling = None

        self._is_valid = False
        self._is_initialized = False

        if stop is None:
            self.stop = start
            self.start = 0

        if step is None:
            self.step = 1

    def __set_initialized(self):
        self._is_valid = not self._is_initialized
        self._is_initialized = True

    @property
    def outer(self):
        self.__set_initialized()
        self._is_outer = True
        return self

    @property
    def inner(self):
        self.__set_initialized()
        self._is_inner = True
        return self

    def tile(self, *args):
        self.__set_initialized()
        self._tiling = args
        return self

    def __iter__(self):
        for i in range(self.start, self.stop, self.step):
            yield i


def _range(start, stop=None, step=None):
    return Range(start, stop, step)

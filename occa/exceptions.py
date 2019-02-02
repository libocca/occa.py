class Error(Exception):
    def __init__(self, message):
        super().__init__(message)


class CError(Error):
    def __init__(self, message):
        super().__init__(message)


class UninitializedError(Error):
    def __init__(self, message):
        super().__init__(message)


class BufferWarning(Warning):
    def __init__(self, message):
        super().__init__(message)

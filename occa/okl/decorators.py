from .kernel import Kernel


def kernel(*args, **kwargs):
    # @okl.kernel
    if (len(args) == 1 and
        len(kwargs) == 0 and
        callable(args[0])):
        return Kernel(args[0])

    # okl.kernel(*args, **kwargs)
    def _kernel(func):
        return Kernel(func, **kwargs)
    return _kernel

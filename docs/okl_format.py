from occa import okl

# Kernel types
@okl.kernel
def foo(a1: int,   # int
        a2: [int], # int*
        a3: okl.dtype(int), # int
        a4: okl.dtype(int, okl.restrict), # int @restrict
        a5: okl.dtype.foo):       # typedef foo
    pass

# @shared / @exclusive types
def foo():
    a = okl.shared(16, dtype=int)

# Loops
@okl.kernel
def foo():
    for i in okl.range(256).tile(16):
        pass

    for block in okl.range(0, 256, 16).outer:
        for i in okl.range(block, block + 16).inner:
            pass

    for block in okl.range(16).outer:
        for i in okl.range(16).inner:
            id = block * 16 + i

# Kernel
@okl.kernel
def foo(a, b, c):
    pass

foo[device](a, b, c, props={})
# or
foo(a, b, c, props={})

def kernel(func):
    # Get David the analogous OKL code from 'func'
    return Kernel(source, name)

def Kernel:
    def __init__(self, source, name):
        self.source = source
        self.name = name

    def build(self, device, props):
        # TODO: If slow, cache kernels
        return device.build_kernel_from_string(self.source,
                                               self.name,
                                               props)

    def __getitem__(self, device):
        return self.build(device)

    def __call__(self, *args, props=None):
        return self.build(occa.get_device(), props=props)(*args)

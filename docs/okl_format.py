from occa import okl

# Kernel types
@okl.kernel
def foo(a1: int,   # int
        a2: [int], # int*
        a3: okl.dtype(int), # int
        a4: okl.dtype(int, okl.restrict), # int @restrict
        a5: okl.dtype.foo):               # typedef foo
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

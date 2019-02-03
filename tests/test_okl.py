import inspect
from typing import List

from occa import okl
from occa.okl.types import Const, Exclusive, Shared


def add(a: Const[float],
        b: Const[float]) -> float:
    return a + b


@okl.kernel
def add_vectors(a: Const[List[float]],
                b: Const[List[float]],
                ab: List[float]) -> None:
    for i in okl.range(len(a)).tile(16):
        foo: float = 2
        s_foo: Shared[List[float, 20, 30]]
        e_foo: Exclusive[float]
        bar: Const[List[float, 2]] = [1,2]
        ab[i] = add(a[i], b[i])


OKL_SOURCE = '''
float add(const float a,
          const float b);

float add(const float a,
          const float b) {
  return a + b;
}

@kernel void add_vectors(const float *a,
                         const float *b,
                         float *ab) {
  for (int i = 0; i < 10; ++i) {
    float foo = 2;
    @shared float s_foo[20][30];
    @exclusive float e_foo;
    const float bar[2] = {1, 2};
    ab[i] = add(a[i], b[i]);
  }
}
'''


def test_okl():
    assert add_vectors.__okl_source__.strip() == OKL_SOURCE.strip()

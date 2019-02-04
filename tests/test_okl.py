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
    for i in range(3):
        pass
    for i in okl.range(len(a)).tile(16):
        # Type tests
        foo: bool = True
        bar: np.float64 = 3.2
        s_foo: Shared[List[float, 20, 30]]
        e_foo: Exclusive[float]
        bar: Const[List[float, 2]] = [1,2]

        s: str = 'string'

        # Flow tests
        for j in range(10):
            pass

        if i > 10:
            continue
        elif i < 5:
            break
        else:
            i: int = 3
            return 3

        while True:
            pass

        # Closure tests
        ab[i] = add(a[i], b[i])


OKL_SOURCE = '''
double add(const double a,
           const double b);

double add(const double a,
           const double b) {
  return a + b;
}

@kernel void add_vectors(const double *a,
                         const double *b,
                         double *ab) {
  for (int i = 0; i < 3; ++i) {}
  for (int i = 0; i < a__len__; ++i; @tile(16, @outer, @inner)) {
    bool foo = true;
    double bar = 3.2;
    @shared double s_foo[20][30];
    @exclusive double e_foo;
    const double bar[2] = {1, 2};
    char *s = "string";
    for (int j = 0; j < 10; ++j) {}
    if (i > 10) {
      continue;
    }
    else if (i < 5) {
      break;
    }
    else  {
      int i = 3;
      return 3;
    }
    while (true) {}
    ab[i] = add(a[i], b[i]);
  }
}

'''


def test_okl():
    assert add_vectors.get_source().strip() == OKL_SOURCE.strip()

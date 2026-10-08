import sympy as sp

i, j, k, n = sp.symbols("i j k n", integer=True, postive=True)
A = sp.MatrixSymbol("A", n, n)
B = sp.MatrixSymbol("B", n, n)

C_ij = sp.Sum(A[i, k] * B[k, j], (k, 1, n))

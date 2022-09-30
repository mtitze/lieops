import numpy as np
import pytest
from scipy.linalg import expm

from lieops.linalg.bch import bch_2x2

def test_bch_2x2(n=6, tol=1e-12):
    for k in range(n):
        A = (1 - 2*np.random.rand(2, 2)) + (1 - 2*np.random.rand(2, 2))*1j
        B = (1 - 2*np.random.rand(2, 2)) + (1 - 2*np.random.rand(2, 2))*1j
        C = bch_2x2(A, B, tol=tol)
        zero = expm(A)@expm(B) - expm(C)
        dim, _ = zero.shape
        assert all([abs(zero[i, j]) < tol for i in range(dim) for j in range(dim)]), f'Run {k} failed.\nA = {A}\nB= {B}'

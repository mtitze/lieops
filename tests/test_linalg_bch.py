import numpy as np
import pytest
from scipy.linalg import expm

from lieops.linalg.bch import bch_2x2
from lieops.linalg.bch.bch_2x2 import get_params

np.random.seed(142)

def bch_2x2_multiplication_table(A, B):
    '''
    Get the multiplication table in Thm. 1 in Ref. [1].

    References
    ----------
    [1] Foulis: "The algebra of complex 2 × 2 matrices and a 
                 general closed Baker–Campbell–Hausdorff formula",
                 J. Phys. A: Math. Theor. 50 305204 (2017).
    '''
    params = get_params(A, B)
    alpha = params['alpha']
    beta = params['beta']
    a = params['a']
    b = params['b']
    epsilon = params['epsilon']
    omega = params['omega']
    tau = params['abs_tau']
    sigma = params['abs_sigma']
    C = params['C']
    I = params['I']
        
    T11 = -alpha*I + a*A
    T12 = ((omega - a*b)*I + b*A + a*B + C)/2
    T13 = ((a*epsilon - 2*b*sigma**2)*I - 2*epsilon*A + 4*sigma**2*B + a*C)/2
    T21 = ((omega - a*b)*I + b*A + a*B - C)/2
    T22 = -beta*I + b*B
    T23 = ((2*a*tau**2 - b*epsilon)*I - 4*tau**2*A + 2*epsilon*B + b*C)/2
    T31 = ((2*b*sigma**2 - a*epsilon)*I + 2*epsilon*A - 4*sigma**2*B + a*C)/2
    T32 = ((b*epsilon - 2*a*tau**2)*I + 4*tau**2*A - 2*epsilon*B + b*C)/2
    T33 = (epsilon**2 - 4*sigma**2*tau**2)*I
    return [[T11, T12, T13], [T21, T22, T23], [T31, T32, T33]]
        
#########
# Tests #
#########

def test_bch_2x2(n=6, tol=1e-12):
    for k in range(n):
        A = (1 - 2*np.random.rand(2, 2)) + (1 - 2*np.random.rand(2, 2))*1j
        B = (1 - 2*np.random.rand(2, 2)) + (1 - 2*np.random.rand(2, 2))*1j
        
        # test the multiplication table given in Ref. [1]
        tab = bch_2x2_multiplication_table(A, B)
        C = A@B - B@A
        reftab = [[A@A, A@B, A@C], [B@A, B@B, B@C], [C@A, C@B, C@C]]
        for i in range(3):
            for j in range(3):
                diff = tab[i][j] - reftab[i][j]
                assert all([abs(diff[k, l]) < tol for k in range(2) for l in range(2)])
                
        # test whether exp(A)@exp(B) is properly combined
        Z = bch_2x2(A, B, tol=tol)
        zero = expm(A)@expm(B) - expm(Z)
        assert all([abs(zero[i, j]) < tol for i in range(2) for j in range(2)]), f'Run {k} failed.\nA = {A}\nB= {B}'
        

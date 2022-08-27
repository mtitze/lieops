import pytest

import numpy as np
from lieops.ops.tools import poly2ad, ad2poly, poly3ad, ad3poly
from lieops import create_coords

def test_poly2ad(n=4, tol=1e-14):
    '''
    Test to check whether the adjoint representation of a 2nd order homogeneous polynomial properly reflects
    the commutator relations.
    '''
    n_const = 10
    q1, q2, p1, p2 = create_coords(2, real=True)
    for k in range(n):
        f = (1 - np.random.rand(n_const)*2) + (1 - np.random.rand(n_const)*2)*1j
        g = (1 - np.random.rand(n_const)*2) + (1 - np.random.rand(n_const)*2)*1j

        h1 = q1**2*f[0] + q2**2*f[1] + p1**2*f[2] + p2**2*f[3] + \
             q1*q2*f[4] + q1*p1*f[5] + q1*p2*f[6] + q2*p1*f[7] + \
             q2*p2*f[8] + p1*p2*f[9]

        h2 = q1**2*g[0] + q2**2*g[1] + p1**2*g[2] + p2**2*g[3] + \
             q1*q2*g[4] + q1*p1*g[5] + q1*p2*g[6] + q2*p1*g[7] + \
             q2*p2*g[8] + p1*p2*g[9]
        
        h12 = h1@h2

        h1mat = poly2ad(h1)
        h2mat = poly2ad(h2)
        h12mat = h1mat@h2mat - h2mat@h1mat
        h12_back = ad2poly(h12mat, tol=tol)

        # check if back and forth transformation are inverses to each other:
        assert (poly2ad(h12_back) - h12mat < tol).all()
        diff = np.array(list((ad2poly(poly2ad(h12), tol=tol) - h12).values()))
        assert (diff < tol).all()

        # check if the commutator is satisfied:
        diff_c = np.array(list((h12_back - h12).values()))
        assert (diff_c < tol).all()

def test_poly3ad(n=4, tol=1e-14):
    '''
    Test to check whether the adjoint representation of a 2nd order polynomial properly reflects
    the commutator relations.
    '''
    n_const = 15
    q1, q2, p1, p2 = create_coords(2, real=True)
    for k in range(n):
        f = (1 - np.random.rand(n_const)*2) + (1 - np.random.rand(n_const)*2)*1j
        g = (1 - np.random.rand(n_const)*2) + (1 - np.random.rand(n_const)*2)*1j

        h1 = q1**2*f[0] + q2**2*f[1] + p1**2*f[2] + p2**2*f[3] + \
             q1*q2*f[4] + q1*p1*f[5] + q1*p2*f[6] + q2*p1*f[7] + \
             q2*p2*f[8] + p1*p2*f[9] + \
             q1*f[10] + q2*f[11] + p1*f[12] + p2*f[13] + f[14]

        h2 = q1**2*g[0] + q2**2*g[1] + p1**2*g[2] + p2**2*g[3] + \
             q1*q2*g[4] + q1*p1*g[5] + q1*p2*g[6] + q2*p1*g[7] + \
             q2*p2*g[8] + p1*p2*g[9] + \
             q1*g[10] + q2*g[11] + p1*g[12] + p2*g[13] + g[14]
        
        h12 = h1@h2
        #h12.pop((0, 0, 0, 0))

        h1mat = poly3ad(h1)
        h2mat = poly3ad(h2)
        h12mat = h1mat@h2mat - h2mat@h1mat
        h12_back = ad3poly(h12mat, tol=tol)

        # check if back and forth transformation are inverses to each other:
        assert (poly3ad(h12_back) - h12mat < tol).all()
        diff = np.array(list((ad3poly(poly3ad(h12), tol=tol) - h12).values()))
        assert (diff < tol).all()

        # check if the commutator is satisfied: poly3ad(h1@h2) = [poly3ad(h1), poly3ad(h2)]
        diff_c = np.array(list((h12_back - h12).values()))
        assert (diff_c < tol).all()


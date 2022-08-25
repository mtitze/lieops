import pytest

import numpy as np
from lieops.ops.tools import poly2ad, ad2poly
from lieops import create_coords

def test_adjoint_repr(n=4, tol=1e-15):
    '''
    Test to check whether the adjoint representation of a 2nd order homogeneous polynomial properly reflects
    the commutator relations.
    '''
    for k in range(n):
        f, g = [], []
        for k in range(10):
            f.append(1 - np.random.rand()*2)
            g.append(1 - np.random.rand()*2)

        q1, q2, p1, p2 = create_coords(2, real=True)
        h1 = f[0]*q1**2 + f[1]*q2**2 + f[2]*p1**2 + f[3]*p2**2 + \
             f[4]*q1*q2 + f[5]*q1*p1 + f[6]*q1*p2 + f[7]*q2*p1 + \
             f[8]*q2*p2 + f[9]*p1*p2

        h2 = g[0]*q1**2 + g[1]*q2**2 + g[2]*p1**2 + g[3]*p2**2 + \
             g[4]*q1*q2 + g[5]*q1*p1 + g[6]*q1*p2 + g[7]*q2*p1 + \
             g[8]*q2*p2 + g[9]*p1*p2
        
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
        
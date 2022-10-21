import pytest

import numpy as np
from lieops.ops.tools import poly2ad, ad2poly, poly3ad, ad3poly, vec2poly, poly2vec
from lieops import create_coords
from lieops.linalg.matrix import create_J

from .common import make_random_cmplx, get_max

def create_sp2n_matrix(dim, max_path=2*np.pi):
    '''
    Create an arbitrary matrix in sp(2n; C), the Lie-algebra of complex
    (2n)x(2n)-matrices.
    
    Parameters
    ----------
    dim: int
        The dimension 'n' of the considered problem.
        
    max_path: float, optional
        An optional parameter to control the extend of the neighbourhood around zero.
        
    Returns
    -------
    ndarray
        An array representing a matrix A with the property A@J + J@A.transpose() = 0.
    
    '''
    A = (1 - 2*np.random.rand(dim, dim)) + 1j*(1 - 2*np.random.rand(dim, dim))
    A = A - A.transpose()
    B = (1 - 2*np.random.rand(dim, dim)) + 1j*(1 - 2*np.random.rand(dim, dim))
    B = B + B.transpose()
    C = (1 - 2*np.random.rand(dim, dim)) + 1j*(1 - 2*np.random.rand(dim, dim))
    C = C + C.transpose()
    return np.block([[A, B], [C, -A.transpose()]])*max_path

#########
# Tests #
#########

def test_poly2ad(n=4, tol=1e-14):
    '''
    Test to check whether the adjoint representation of a 2nd order homogeneous polynomial properly reflects
    the commutator relations.
    '''
    n_const = 10
    q1, q2, p1, p2 = create_coords(2, real=True)
    for k in range(n):
        f = make_random_cmplx(n_const)
        g = make_random_cmplx(n_const)

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
        assert (np.abs(poly2ad(h12_back) - h12mat) < tol).all()
        diff = np.array(list((ad2poly(poly2ad(h12), tol=tol) - h12).values()))
        assert (np.abs(diff) < tol).all()

        # check if the commutator is satisfied:
        diff_c = np.array(list((h12_back - h12).values()))
        assert (np.abs(diff_c) < tol).all()

def test_poly3ad(n=4, tol=1e-14):
    '''
    Test to check whether the adjoint representation of a 2nd order polynomial properly reflects
    the commutator relations.
    '''
    n_const = 15
    q1, q2, p1, p2 = create_coords(2, real=True)
    for k in range(n):
        f = make_random_cmplx(n_const)
        g = make_random_cmplx(n_const)

        h1 = q1**2*f[0] + q2**2*f[1] + p1**2*f[2] + p2**2*f[3] + \
             q1*q2*f[4] + q1*p1*f[5] + q1*p2*f[6] + q2*p1*f[7] + \
             q2*p2*f[8] + p1*p2*f[9] + \
             q1*f[10] + q2*f[11] + p1*f[12] + p2*f[13]

        h2 = q1**2*g[0] + q2**2*g[1] + p1**2*g[2] + p2**2*g[3] + \
             q1*q2*g[4] + q1*p1*g[5] + q1*p2*g[6] + q2*p1*g[7] + \
             q2*p2*g[8] + p1*p2*g[9] + \
             q1*g[10] + q2*g[11] + p1*g[12] + p2*g[13]
        
        h12 = h1@h2
        h12_no_const = h12.pop((0, 0, 0, 0), None)
        
        h1mat = poly3ad(h1)
        h2mat = poly3ad(h2)
        h12mat = h1mat@h2mat - h2mat@h1mat
        h12_back = ad3poly(h12mat, tol=tol)

        # check if back and forth transformation are inverses to each other:
        assert (np.abs(poly3ad(h12_back) - h12mat) < tol).all()
        diff = np.array(list((ad3poly(poly3ad(h12_no_const), tol=tol) - h12_no_const).values()))
        assert (np.abs(diff) < tol).all()

        # check if the commutator is satisfied: poly3ad(h1@h2) = [poly3ad(h1), poly3ad(h2)]
        diff_c = np.array(list((h12_back - h12_no_const).values()))
        assert (np.abs(diff_c) < tol).all()

@pytest.mark.parametrize("dim", [1, 1, 2, 2, 3, 3])
def test_vec2poly(dim, tol=1e-12):
    '''
    Test if matrix multiplication and matrices applied to vectors are compatible with
    the vec2poly, poly2vec, ad2poly and poly2ad routines, using random sp(2n; C)-matrices.
    '''
    
    R1 = create_sp2n_matrix(dim)
    R2 = create_sp2n_matrix(dim)
    JJ = create_J(dim)
    R1p = ad2poly(R1)
    R2p = ad2poly(R2)

    zero1m = R1.transpose()@JJ + JJ@R1
    zero2m = R2.transpose()@JJ + JJ@R2
    zero3m = poly2ad(R1p) - R1
    zero4m = poly2ad(R2p) - R2
    for zerom in [zero1m, zero2m, zero3m, zero4m]:
        assert (np.abs(zerom) < tol).all()

    z = make_random_cmplx(dim*2)
    zp = vec2poly(z)
    
    v1p = R1p@zp
    v1 = R1@z    
    v2p = R2p@v1p
    v2 = R2@v1
    
    zero1p = vec2poly(v1) - v1p
    zero1v = poly2vec(v1p) - v1
    zero2p = vec2poly(v2) - v2p
    zero2v = poly2vec(v2p) - v2
    
    assert get_max(zero1p) < tol
    assert get_max(zero2p) < tol
    for zerov in [zero1v, zero2v]:
        assert (np.abs(zerov) < tol).all()
    
    # Check equation "{A, z_j} = A_{ij} z_i"
    xieta = create_coords(dim)
    # right-hand side
    xietaf = []
    for j in range(dim*2):
        xietaf.append(sum([xieta[i]*R1[i, j] for i in range(dim*2)]))
    # left-hand side
    xietaf2 = []
    for c in xieta:
        xietaf2.append(R1p@c)
    # equality check 
    for k in range(dim*2):
        assert get_max(xietaf[k] - xietaf2[k]) < tol
        
    # check composition of matrices (again):
    assert (np.abs(poly2vec(R1p@(R2p@zp)) - R1@R2@z) < tol).all()
    # ... as well as the commutator:
    assert (np.abs(poly2vec((R1p@R2p)@zp) - (R1@R2 - R2@R1)@z) < tol).all()


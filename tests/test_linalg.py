import numpy as np
import pytest

from lieops.linalg.matrix import create_J
from lieops.linalg.diagonalize import lemma9, thm10, cor29
from lieops.linalg.nf import _identifyPairs

from scipy.linalg import expm

def create_spn_matrix(dim, max_path=2*np.pi, tol=0):
    r'''
    Create an element of Sp(n) = Sp(2n; C) \cap U(2n), the compact and simply connected group of
    symplectic and unitary matrices. 
    
    Parameters
    ----------
    dim: int
        Controls the dimension of the requested matrix. A matrix of shape
        (2*dim)x(2*dim) will be returned.
    
    max_path: float, optional
        A scaling factor to traverse the lie-algebra.
        
    tol: float, optional
        An optional tolerance to perform basic checks, if a value > 0 is provided.
        
    Returns
    -------
    S: array-like
        The respective element in the Lie-group Sp(2n).
        
    X: array-like
        A (2*dim)x(2*dim) matrix within the Lie-algebra of Sp(2n). It holds exp(X) = S.
    '''
    # Step 1: Construct a random unitary symplectic matrix
    P = (1 - 2*np.random.rand(dim, dim)) + (1 - 2*np.random.rand(dim, dim))*1j
    P = P + P.transpose()
    Q = (1 - 2*np.random.rand(dim, dim)) + (1 - 2*np.random.rand(dim, dim))*1j
    Q = Q + -Q.transpose().conj()
    X = np.block([[Q, P], [-P.conj(), Q.conj()]])*max_path
    
    # now use the fact that exp is surjective onto Sp(n):
    S = expm(X)

    if tol > 0:
        dim2 = 2*dim
        J = np.array(create_J(dim)).transpose()
        zero1 = X.transpose()@J + J@X # X in sp(2n; C)
        zero2 = X.transpose().conj() + X # X in u(2n)
        zero3 = S.transpose()@J@S - J # S in Sp(2n; C)
        zero4 = S.transpose().conj()@S - np.eye(dim2) # S in U(2n)
        for zero in [zero1, zero2, zero3, zero4]:
            assert all([abs(zero[i, j]) < tol for i in range(dim2) for j in range(dim2)])  
    
    return S, X

#########
# Tests #
#########

@pytest.mark.parametrize("dim", [2, 2, 4, 4, 6, 6])
def test_cor29(dim, tol=1e-14, **kwargs):
    '''
    Test Corollary 29 for the special normal and (skew)-Hamiltonian (J-(skew)-symmetric) matrices
    M +- J@M.transpose()@J ,
    where M is an element in Sp(n) (i.e. a special normal and J-normal matrix.)
    '''
    W, _ = create_spn_matrix(dim, tol=tol, **kwargs)
    U = cor29(W)
    
    J = np.array(create_J(dim)).transpose()
    zero1 = U.transpose().conj()@U - np.eye(dim*2)
    zero2 = U.transpose()@J@U - J
    for zero in [zero1, zero2]:
        assert all([abs(zero[i, j]) < tol for i in range(dim*2) for j in range(dim*2)])
    
    diag = U.transpose().conj()@W@U
    for j in range(dim*2):
        assert all([abs(diag[i, j]) < tol and abs(diag[j, i]) < tol for i in range(j)])
    D = diag.diagonal()
    pairs = _identifyPairs(D, condition=lambda a, b: abs(a - b.conj()) < tol)
    assert len(pairs) == dim
    
    
    
    
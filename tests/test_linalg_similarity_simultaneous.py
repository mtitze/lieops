import pytest
import numpy as np

from lieops.linalg.similarity import simuldiag

# Preliminary functions to create the tests.

def matrix_power(X, k: int):
    '''
    Compute the power of a matrix.
    '''
    out = X
    for j in range(k):
        out = X@out
    return out

def create_normal_commuting_matrices(dim: int, power: int=3, tol=0):
    '''
    Create two random normal complex square matrices X and Y which commute.
    
    This means they admit the following properties:
    1) X@Y = Y@X
    2) X@X^H = X^H@X
    3) Y@Y^H = Y^H@Y
    
    Parameters
    ----------
    dim: int
        The dimension of the matrices.
        
    power: int
        The second matrix Y is given in terms of a polynomial of X up to a specific power.
        
    tol: float, optional
        An optional tolerance to perform consistency checks.
        
    Returns
    -------
    X: ndarray
        The first matrix.
        
    Y: ndarray
        The second matrix.
    '''
    XX = (1 - 2*np.random.rand(dim, dim)) + (1 - 2*np.random.rand(dim, dim))*1j
    XX = XX@XX.conj().transpose()
    YY = sum([(1 - 2*np.random.rand(1) + (1 - 2*np.random.rand(1))*1j)[0]*matrix_power(XX, j) for j in range(power)])

    if tol > 0:
        # consistency checks; if 'power' has been chosen high, round-off errors may become significant.
        zero0 = XX@XX.conj().transpose() - XX.conj().transpose()@XX
        zero1 = XX@YY - YY@XX
        zero2 = YY@YY.conj().transpose() - YY.conj().transpose()@YY
        for zero in [zero0, zero1, zero2]:
            assert all([abs(zero[i, j]) < tol for i in range(dim) for j in range(dim)]), f'{zero}' 
        
    return XX, YY

#########
# Tests #
#########

@pytest.mark.parametrize("dim", [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
def test_simuldiag(dim, tol1=5e-10, tol2=1e-15, **kwargs):
    '''
    Test simuldiag (simultaneous diagonalization of normal matrices) routine in various dimensions.
    '''
    XX, YY = create_normal_commuting_matrices(dim=dim, tol=tol1, **kwargs)
    Q, A, err, n_sweeps = simuldiag(XX, YY, tol=tol2, **kwargs)
    
    assert err < tol1
    zero0 = Q@Q.transpose().conj() - np.eye(dim)
    D1 = Q@XX@Q.transpose().conj()
    zero1 = D1 - np.diag(D1.diagonal())
    D2 = Q@YY@Q.transpose().conj()
    zero2 = D2 - np.diag(D2.diagonal())

    for zero in [zero0, zero1, zero2]:
        assert all([abs(zero[i, j]) < tol1 for i in range(dim) for j in range(dim)])

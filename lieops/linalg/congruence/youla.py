import numpy as np
import cmath

from lieops.linalg.common import twonorm, basis_extension
from lieops.linalg.checks import _check_linear_independence

from .common import rotate_2block

def skew_post_youla(M):
    r'''
    A skew-symmetric complex matrix M in Youla normal form will admit 2x2 blocks of the form
    
          / 0    x \
      B = |        |
          \ -x   0 /
          
    This routine will determine an additional unitary matrix U so that U.transpose()@M@U will be in
    the same block form, but the entries x will be real and non-negative.
    
    Parameters
    ----------
    M:
        Complex skew-symmetric matrix in Youla normal form.
        
    Returns
    -------
    U:
        Complex unitary matrix as described above.
    '''
    dim = len(M)
    assert dim%2 == 0
    U = np.eye(dim, dtype=complex)
    for k in range(dim//2):
        k2 = 2*k
        x = M[k2, k2 + 1]
        U[k2: k2 + 2, k2: k2 + 2] = rotate_2block(x)
    return U

def youla_normal_form(M, tol=1e-13, **kwargs):
    '''
    Transform a matrix into Youla normal form according to Ref. [1]:

    Y = U.transpose()@M@U.
    
    Statement of the theorem (see Ref. [1] for the notation):
    Any complex square matrix M can be brought by a unitary congruence transformation to a block triangular 
    form with the diagonal blocks of orders 1 and 2. The 1×1 blocks correspond to real 
    nonnegative coneigenvalues of M, while each 2×2 block corresponds to a pair of complex 
    conjugate coneigenvalues.
    
    N.B. If it appears that the matrix U does not transform into the desired form, try a change in the tolerance parameter.
    
    Parameters
    ----------
    M:
        Matrix to be transformed.
        
    tol:
        Tolerance passed to ._check_linear_independence routine.
        
    Returns
    -------
    U:
        Unitary matrix so that U.transpose()@M@U is in Youla normal form.
        
    References
    ----------
    [1] H. Faßbender and Kh. D. Ikramov: "Some observations on the Youla form and conjugate-normal matrices" (2006).

    '''
    dim = len(M)
    if dim == 0:
        return np.zeros([0, 0])
    elif dim == 1:
        return np.eye(1)
    
    # Get an eigenvector. TODO: Perhaps there is a faster way (similar to QR-algorithm for the Schur decomposition).
    
    # 1. option: this step seems to be not repeatable; always a different value is returned.
    #from scipy.sparse.linalg import eigs
    #ev, x1 = eigs(M.conjugate()@M, k=1)
    #x1 = np.array(x1.transpose().tolist()[0])
    #print ('check:', M.conjugate()@M@x1 - ev*x1, 'ev:', ev)
    
    # 2. option use np.linalg.eig:
    eigenvalues, eigenvectors = np.linalg.eig(M.conjugate()@M)
    x1 = np.array(eigenvectors[:, 0]).flatten()
    #print ('check:', M.conjugate()@M@x1 - eigenvalues[0]*x1, 'ev:', eigenvalues)
    
    x2 = np.array(M@x1).flatten().conjugate()
    U = np.zeros([dim, dim], dtype=complex)
    if _check_linear_independence(x1, x2, tol=tol):
        u1 = x1/twonorm(x1)
        u2 = x2 - (u1.transpose().conjugate()@x2)*u1
        u2 = u2/twonorm(u2)
        ext = basis_extension(u1, u2, gs=True)
        U[:, 0] = u1
        U[:, 1] = u2
        k = 2
    else:
        u1 = x1/twonorm(x1)
        ext = basis_extension(u1, gs=True)
        U[:, 0] = u1
        k = 1
    U[:, k:] = ext
    
    M_youla = U.transpose()@M@U
    
    U_submatrix = np.zeros([dim, dim], dtype=complex)
    U_submatrix[:k, :k] = np.eye(k)
    U_submatrix[k:, k:] = youla_normal_form(M_youla[k:, k:], tol=tol, **kwargs)
    return U@U_submatrix

def unitary_anti_diagonalize_skew(M, tol=1e-14, **kwargs):
    r'''Anti-diagonalize a complex skew symmetric matrix M,
    so that U.transpose()@M@U is skew-symmetric anti-diagonal with respect 
    to (n//2 x n//2) block matrices of the form
            
             /  0   X  \
             |         |
             \ -X   0  /
    
    where X is a diagonal matrix with positive entries. This is accomplished by Youla-decomposition.
    
    Parameters
    ----------
    M:
        list of vectors defining a real skew symmetric (n x n)-matrix.
    
    Returns
    -------
    U:
        Unitary complex matrix 
    '''
    assert all([abs((M.transpose() + M)[j, k]) < tol for j in range(len(M)) for k in range(len(M))]), f'Matrix not anti-symmetric within given tolerance {tol}.'
    U = youla_normal_form(M, **kwargs)
    My = U.transpose()@M@U
    U1 = skew_post_youla(My)
    return U@U1

def unitary_diagonalize_symmetric(M, tol=1e-14, **kwargs):
    '''
    Compute a unitary matrix U so that U.transpose()@M@U =: D is diagonal with real non-zero entries (Autonne & Takagi).
    
    Parameters
    ----------
    M:
        Matrix to be diagonalized.
        
    tol: float, optional
        A tolerance parameter by which the given matrix is checked for symmetry
        
    **kwargs
        Additional parameters passed to .youla_normal_form
        
    Returns
    -------
    U:
        Unitary matrix U with the above property.
    '''
    assert all([abs((M.transpose() - M)[j, k]) < tol for j in range(len(M)) for k in range(len(M))]), f'Matrix not symmetric within given tolerance {tol}.'
    U = youla_normal_form(M, **kwargs)
    D1 = U.transpose()@M@U
    # now turn the complex diagonal values of D1 into real values
    U1 = np.diag([np.exp(-1j*cmath.phase(D1[k, k])/2) for k in range(len(M))])
    return U@U1


# This script will collect various checks related to Linear Algebra.

import numpy as np
import mpmath as mp

from njet.functions import get_package_name

def _check_linear_independence(a, b, tol=1e-14):
    '''
    Check if two vectors are linearly independent.
    It is assumed that they are both non-zero.
    
    Parameters
    ----------
    a: subscriptable
        The first vector to be checked.
        
    b: subscriptable
        The second vector to be checked.
        
    tol: float, optional
        A tolerance below which we consider values to be equal to zero.
        
    Returns
    -------
    boolean
        If True, then both vectors appear to be linearly independent.
    '''
    assert len(a) == len(b)
    dim = len(a)
    q = 0
    for k in range(dim):
        if (abs(a[k]) < tol and abs(b[k]) > tol) or (abs(a[k]) > tol and abs(b[k]) < tol):
            return True 
        elif abs(a[k]) < tol and abs(b[k]) < tol:
            continue
        else: # both a[k] and b[k] != 0
            qk = a[k]/b[k]
            if abs(q) > tol and abs(q - qk) > tol:
                return True
            q = qk
    return False

def is_positive_definite(A):
    '''Check if a given matrix A is positive definite.

    Parameters
    ----------
    A: matrix
        The matrix to be checked.
        
    code: str, optional
        The code to be used for the check. Either 'mpmath' or 'numpy' (default).
        
    Returns
    -------
    boolean
        True if matrix is positive definite.
    '''
    code = get_package_name(A)
    
    out = True
    if code == 'mpmath':
        try:
            _ = mp.cholesky(A)
        except:
            out = False
    else:
        try:
            _ = np.linalg.cholesky(A)
        except:
            out = False
    return out
    
def relative_eq(a, b, tol=1e-14, **kwargs):
    '''
    Check if the relative difference between two values is smaller than a given value tol.
    
    Parameters
    ----------
    a: complex
        First parameter
    
    b: complex
        Second parameter
        
    tol: float, optional
        Tolerance below which we consider the relative difference of the two parameters as 
        zero.
        
    Returns
    -------
    boolean
    '''
    if a == 0 and b == 0:
        return True
    else:
        return abs(a - b)/max([abs(a), abs(b)]) < tol
    
def williamson_check(A, S, J, tol=1e-14):
    '''
    Check if a given matrix A and the matrix S diagonalizting A according to the theorem of Williamson
    by S.transpose()*A*S = D actually are satisfying all required conditions of the theorem.
    
    If any condition is violated, the routine raises an AssertionError.
    
    Parameters
    ----------
    A: matrix
        The matrix to be diagonalized.
        
    S: matrix
        The symplectic matrix obtained by Williamson's theorem.
        
    code: str, optional
        The code by which the check should be performed. Either 'mpmath' or 'numpy' (default).
        
    tol: float, optional
        A tolerance by which certain properties (like matrix entries) are considered to be zero (default 1e-14).
    '''
    code = get_package_name(A)

    if code == 'numpy':
        isreal = np.isreal(A).all()
        isposdef = is_positive_definite(A, code=code)
        issymmetric = np.all(A - A.transpose()) == 0
        isevendim = len(A)%2 == 0
        symplecticity = np.linalg.norm(S.transpose()*J*S - J)
        issymplectic = symplecticity < tol
        
        diag = J@S@J@A@J@S.transpose()@J
        offdiag = np.array([[diag[k, l] if k != l else 0 for k in range(len(diag))] for l in range(len(diag))])
        isdiagonal = np.all(np.abs(offdiag) < tol)

    elif code == 'mpmath':
        isreal = mp.norm(A - A.conjugate()) == 0
        isposdef = is_positive_definite(A, code=code)
        issymmetric = all([[(A - A.transpose())[i, j] == 0 for i in range(len(A))] for j in range(len(A))])
        isevendim = len(A)%2 == 0
        symplecticity = mp.norm(S.transpose()@J@S - J)
        issymplectic = symplecticity < tol
        
        diag = J@S@J@A@J@S.transpose()@J
        absoffdiag = np.array([[abs(complex(diag[k, l])) if k != l else 0 for k in range(len(diag))] for l in range(len(diag))])
        isdiagonal = np.all(absoffdiag < tol)
        
    assert isreal, 'Input matrix A not real.'
    assert isposdef, 'Input matrix A not positive definite.'
    assert issymmetric,  'Input matrix A not symmetric.'
    assert isevendim, 'Dimension not even.'
    assert issymplectic, f'Symplecticity not ensured: |S^(tr)@J@S - J| = {symplecticity} >= {tol} (tol)'
    assert isdiagonal, f'Matrix D = S^(-tr)@M@S^(-1) =\n{diag}\nappears not to be diagonal (one entry having abs > {tol} (tol)).'


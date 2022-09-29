import numpy as np
import mpmath as mp

from sympy import Matrix as sympy_matrix
from sympy import diag as sympy_diag

from njet.functions import get_package_name

from lieops.linalg.matrix import create_J
from lieops.linalg.common import eigenspaces, get_principal_sqrt
from lieops.linalg.similarity.common import diagonal2block
from lieops.linalg.matrix import expandingSum

def symplectic_takagi_old(G, d2b_tol=1e-10, check=True, **kwargs):
    '''
    Symplectic Takagi factorization (OLD ROUTINE).
    
    Let G be a symmetric matrix with complex entries and J the canonical symplectic
    structure. Assumes that G@J is diagonalizable.
    
    Then this routine will determine a symplectic matrix S so that S@D@S.transpose() = G, where
    S will be complex in general and D a diagonal matrix.
    
    Parameters
    ----------
    G: matrix
        Symmetric matrix.
        
    d2b_tol: float, optional
        Optional tolerance given to 'diagonal2block' routine.
        
    **kwargs: optional
        Optional arguments passed to 'eigenspaces' routine.
        
    Returns
    -------
    S: matrix
        Symplectic matrix.
        
    D: matrix
        Diagonal matrix.
        
    X: matrix
        Matrix which has been found to anti-diagonalize G@J so that X^(-1)@G@J@X = F is in anti-diagonal form.
    '''
    code = get_package_name(G)
    
    dim2 = len(G)
    assert dim2%2 == 0, 'Dimension must be even.'
    dim = dim2//2
    assert max([max([abs(G[j, k] - G[k, j]) for j in range(dim2)]) for k in range(dim2)]) < kwargs.get('tol', d2b_tol), 'Input matrix does not appear to be symmetric.'
    
    J = create_J(dim)
    if code == 'mpmath':
        J = mp.matrix(J)
    
    GJ = G@J
    
    # Step 1: Anti-block-diagonalize GJ
    evals, evects = eigenspaces(GJ, flatten=True, **kwargs)
    if code == 'numpy':
        Y = np.array(evects).transpose()
        Yi = np.linalg.inv(Y)
    if code == 'mpmath':
        Y = mp.matrix(evects).transpose()
        Yi = mp.inverse(Y)

    # Yi@GJ@Y will be diagonal
    U = diagonal2block(evals, code=code, tol=d2b_tol, orientation=kwargs.get('orientation', []))
    Xi = U.transpose().conjugate()@Yi
    X = Y@U
    F = Xi@GJ@X # F = A and GJ = B in Cor. 5.6
    D = -F@J
    
    # Step 2: Construct symplectic matrix
    if code == 'numpy':
        # from scipy.linalg import sqrtm
        # YY = sqrtm(-J@X.transpose()@J@X) # does not give poylnomial square roots
        # YY = get_principal_sqrt(-J@X.transpose()@J@X) # does not give polynomial square roots in general TODO: need to change this routine
        
        Jmp = mp.matrix(J)
        Ximp = mp.matrix(Xi)
        YY = np.array(mp.sqrtm(-Jmp@Ximp.transpose()@Jmp@Ximp).tolist(), dtype=np.complex128)
    if code == 'mpmath':
        YY = mp.sqrtm(-J@Xi.transpose()@J@Xi)
        
    # It must hold: -J@YY.transpose()@J = Y. If this is not the case, then YY is not a polynomial square root of the above matrix.
    # (see Thm. 5.5 my notes, or Horner: Topics in Matrix Analysis, S-polar decomposition. Throw me a message if you need details)
    if kwargs.get('check', True):
        assert max(np.abs(np.array(-J@YY.transpose()@J - YY, dtype=np.complex128).flatten())) < d2b_tol, 'It appears that the routine to compute the matrix square root does not give a *polynomial* square root.'
        
    return YY@X, D, X

def symplectic_takagi(G, **kwargs):
    '''
    Symplectic Takagi factorization.
    
    Let G be a symmetric matrix with complex entries and J the canonical symplectic
    structure. Assumes that G@J@G@J is diagonalizable.
    
    Then this routine will determine a symplectic matrix S so that S@G@S.transpose() is diagonal.
    
    Parameters
    ----------
    G: matrix
        Symmetric matrix.

    **kwargs: optional
        Optional arguments passed to 'diagonal2block' routine.
        
    Returns
    -------
    S: matrix
        Symplectic matrix.
    '''
    
    dim2 = len(G)
    assert dim2%2 == 0, 'Dimension must be even.'
    dim = dim2//2

    # use sympy to get the jordan normal form
    J = create_J(dim)
    GJs = sympy_matrix(G@J)
    P, cells = GJs.jordan_cells() # P.inv()@GJ@P = JNF
    JNF = sympy_diag(*cells)
    P = np.array(P).astype(np.complex128)
    
    U2 = np.array([[1j, -1j], [1, 1]])/np.sqrt(2) # diagonalizes a 2x2 off-diagonal matrix A via U2.transpose().conjugate()@A@U2
    zero2 = np.zeros([2, 2])
    one2 = np.eye(2)
    
    # identify the 2x2 jordan blocks
    skip = []
    k = 0
    for c in cells:
        if c.shape[0] == 2:
            skip.append(k)
            skip.append(k + 1)
    diagonal_of_JNF = np.array(JNF.diagonal()).astype(np.complex128)[0]
    U = diagonal2block(diagonal_of_JNF, code='numpy', skip=skip, orientation=kwargs.get('orientation', []))
    # The unitary U will anti-diagonalize the diagonal blocks on the JNF via U.transpose().conjugate()@JNF@U.
    
    # Construct an orthogonal T which will serve as a similarity transformation between the direct sum
    # and the expanding sum (see docs in expandingSum for details).
    T = expandingSum(dim)
    UT = U@T # UT.transpose().conjugate()@JNF@UT will be in desired anti-diagonal form
    X = P@UT # This matrix is the desired similarity transform between GJ and the J-invariant anti-diagonal matrix
    Xi = np.linalg.inv(X)

    # Now construct the symplectic diagonalizing matrix Y@Xi: 
    Y = get_principal_sqrt(-J@X.transpose()@J@X)
    if kwargs.get('check', True):
        assert max(np.abs(np.array(-J@Y.transpose()@J - Y, dtype=np.complex128).flatten())) < kwargs.get('d2b_tol', 1e-10), 'It appears that the routine to compute the matrix square root does not give a *polynomial* square root.'
    S = Y@Xi
    return S, S@G@S.transpose()


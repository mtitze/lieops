import numpy as np
import mpmath as mp

from .symmetric import cortho_diagonalize_symmetric
from lieops.linalg.matrix import create_J
from lieops.linalg.similarity.common import diagonal2block, anti_diagonalize_real_skew
from lieops.linalg.common import cortho_symmetric_decomposition

from njet.functions import get_package_name

def williamson(V, **kwargs):
    r'''
    Compute Williamson's decomposition of a symmetric positive definite real matrix,
    according to 'R. Simon, S. Chaturvedi, and V. Srinivasan: Congruences and Canonical Forms 
    for a Positive Matrix: Application to the Schweinler-Wigner Extremum Principle'.
    
    The output matrix S and the diagonal D are satisfying the relation:
    S.transpose()@D@S = V
    
    Attention: No extensive checks if V is actually symmetric positive definite real.
    
    Parameters
    ----------
    V: list
        List of vectors (subscriptables) defining the matrix.
        
    **kwargs
        Additional arguments are passed to .anti_diagonalize_real_skew routine.
    
    Returns
    -------
    S: matrix
        Symplectic matrix with respect to the standard block-diagonal form
        
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
        which diagonalizes V as described above.
        
        Note that results for a different J' can be obtained by applying a matrix congruence operation T.transpose()*S*T to the
        result S, where S is obtained by an input matrix V' by the respective inverse congruence operation on V.
        Here T denotes the transformation satisfying J' = T.transpose()@J@T. 
        
    D: matrix
        The diagonal matrix as described above.    
    '''
    dim2 = len(V)
    assert dim2%2 == 0
    dim = dim2//2
    
    code = get_package_name(V)

    if code == 'numpy':
        evalues, evectors = np.linalg.eigh(V)
        sqrtev = np.sqrt(evalues)
        diag = np.diag(sqrtev)
        diagi = np.diag(1/sqrtev)
        evectors = np.array(evectors)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32)
        V = mp.matrix(V)
        evalues, evectors = mp.eigh(V)
        diag = mp.diag([mp.sqrt(e) for e in evalues])
        diagi = mp.diag([1/mp.sqrt(e) for e in evalues])

    assert all([e > 0 for e in evalues]), f'Eigenvalues of input matrix\n{V}\nnot all positive.'
    
    V12 = evectors@diag@evectors.transpose() # V12 means V^(1/2), the square root of V.
    V12i = evectors@diagi@evectors.transpose()
        
    J = create_J(dim)
    if code == 'mpmath':
        J = mp.matrix(J)
    skewmat = V12i@J@V12i
    A = anti_diagonalize_real_skew(skewmat, **kwargs)    
    K = A.transpose()@skewmat@A # the sought anti-diagonal matrix

    # obtain D as described in the reference above
    Di_values = [K[i, i + dim] for i in range(dim)]*2
    D = [1/e for e in Di_values]
    if code == 'numpy':
        D12i = np.array(np.diag([np.sqrt(e) for e in Di_values]))
        D = np.array(np.diag(D))
    if code == 'mpmath':
        D12i = mp.matrix(mp.diag([mp.sqrt(e) for e in Di_values]))
        D = mp.matrix(mp.diag(D))
    S = D12i@A.transpose()@V12
    return S, D


def unitary_williamson(M, tol=1e-14, **kwargs):
    r'''
    Transform a symmetric invertible diagonalizable matrix M (which may be complex) 
    to a complex diagonal form, by means of a 
    complex symplectic matrix S: 
       S.transpose()@M@S = D.
       
    Note that symplecticity of S means:
       S.transpose()@J@S = J.
    
    Background: This routine can be understood as a 'generalization' of Williamson's Theorem to the case 
    of arbitrary symmetric invertible diagonalizable matrices. I.e. matrices which are not necessarily
    real or positive definite.
    
    Parameters
    ----------
    M:
        A complex symmetric diagonalizable and invertible matrix.
        
    tol: float, optional
        A tolerance parameter by which certain entries are compared for equality.
        
    Returns
    -------
    S:
        A complex symplectic matrix as described in the text above.
        
    K:
        The diagonal matrix K = S.transpose()@M@S.
    '''
    assert all([abs((M.transpose() - M)[j, k]) < tol for j in range(len(M)) for k in range(len(M))]), f'Matrix not symmetric within given tolerance {tol}.'
    dim2 = len(M)
    assert dim2%2 == 0
    dim = dim2//2
    
    # Step 1: Determine a square root of M (Since M is orthogonal diagonalizable, this square root will be is symmetric).
    Y = cortho_diagonalize_symmetric(M, tol=tol)
    DM = Y.transpose()@M@Y
    M12i = Y@np.diag([1/np.sqrt(DM[k, k]) for k in range(dim2)])@Y.transpose()
    # alternative code to compute the inverse square root of M using scipy:
    # from scipy.linalg import sqrtm
    #M12 = sqrtm(M)
    #M12i = np.linalg.inv(M12)
    
    # Step 2: As in the positive definite scenario, consider the anti-symmetric matrix A := M12i@J@M12i. This anti-symmetric matrix A
    # is now complex and we will assume here that it can be diagonalized. 
    # If b is an eigenvalue of A, then 0 = det(A - b) = det(A.transpose() - b) = det(-A - b) = (-1)^(2n) det(A + b),
    # so -b must also be an eigenvalue of A. Therefore the diagonalization of A will admit pairs (+/- b) of eigenvalues on its main diagonal.
    J = create_J(dim) # the default block symplectic structure 
    A = M12i@J@M12i
    
    EV, ES = np.linalg.eig(A)
    ESi = np.linalg.inv(ES)
    DD = ESi@A@ES # DD will be diagonal with the Krein-pairs of eigenvalues described above on its main diagonal.
    U = diagonal2block(DD.diagonal(), code='numpy', tol=tol) # The unitary matrix U will transform DD into block-diagonal form with diagonal entries on the anti-diagonal via U.transpose()@DD@U.
    
    # Step 3: The routine "diagonal2block" internally determines the ordering of the Krein pairs. Since we require one representant for each pair, we compute the block-anti-diagonal result:
    K = U.transpose().conjugate()@DD@U # Note that the additional ".conjugate()" will transform DD into block-anti-diagonal form, which will then be matrix-similar to J.
    assert all([abs((K.transpose() + K)[j, k]) < tol for j in range(len(K)) for k in range(len(K))]), f'Matrix expected to be anti-symmetric within given tolerance {tol}.'
    Li = np.diag([1/np.sqrt(K[i, i + dim]) for i in range(dim)]*2)
    # with L = np.linalg.inv(Li)
    # check L@J@L - K = 0
    
    # Step 4: Since K is similar to A by means of ES@U, there must be a complex orthogonal transformation, mapping A to K (see Corollary 6.4.18 in
    # Horn & Johnson: Topics in Matrix Analysis, 1991).
    SS = U.transpose().conjugate()@ESi
    QQ, GG = cortho_symmetric_decomposition(SS)
    # QQ is the sought complex orthogonal transformation. Now it holds (check):
    # GG@QQ = SS
    # QQ.transpose()@QQ = 1
    # GG.transpose() = GG
    # QQ@A@QQ.transpose() = K

    # Step 5: Now we can construct a complex symplectic transformation S which will congruent-diagonalize the given M
    S = M12i@QQ.transpose()@Li
    # check
    # S.transpose()@J@S = J
    # S.transpose()@M@S = Li@Li
    # N.B. U.conjugate()@S.transpose()@M@S@U.conjugate().transpose() will be in normal form.
    return S, K, U


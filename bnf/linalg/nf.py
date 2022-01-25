# This file collects various algorithms to transform a given matrix into a specific normal form -- or to
# decompose the matrix into a product of other matrices with various properties.

import numpy as np
import mpmath as mp
import cmath

from .tools import basis_extension, eigenspaces, get_principal_sqrt, twonorm
from .checks import is_positive_definite, relative_eq
from .matrix import column_matrix_2_code, create_J
    

def _check_linear_independence(a, b, tol=1e-14):
    '''
    A quick routine to check if two vectors a and b are linearly independent.
    It is assumed that a and b are both non-zero.
    
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


def youla_normal_form(M, tol=1e-13, **kwargs):
    '''
    Transform a matrix into Youla normal form according to Ref. [2]:
    "Some observations on the Youla form and conjugate-normal matrices" from
    H. Faßbender and Kh. D. Ikramov (2006).
    
    Statement of the theorem (see Ref. [2] for the definitions of the names):
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


def _rotate_2block(x):
    r'''
    Helper function to "rotate" a 2x2-matrix M of the form
    
        / 0    x \
    M = |        |
        \ -x   0 /
        
    by means of a unitary matrix U so that U.transpose()@M@U has the same form as M, but
    where which x has no imaginary part and is >= 0.
    
    Parameters
    ----------
    x:
        The entry in the top right corner of M.
    
    Returns
    -------
    U:
        Unitary 2x2 matrix with the property as described above.
    '''
    phi = -cmath.phase(x)
    return np.array([[np.exp(1j*phi/2), 0], [0, np.exp(1j*phi/2)]])


def _skew_post_youla(M):
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
        U[k2: k2 + 2, k2: k2 + 2] = _rotate_2block(x)
    return U


def unitary_anti_diagonalize_skew(M, tol=1e-14, **kwargs):
    r'''Anti-diagonalize a complex skew symmetric matrix M so that it will have the block-form
    
             /  0   X  \
        M =  |         |
             \ -X   0  /
    
    where X is a diagonal matrix with positive entries. This is accomplished by Youla-decomposition.
    
    Parameters
    ----------
    M:
        list of vectors defining a real skew symmetric (n x n)-matrix.
    
    Returns
    -------
    U:
        Unitary complex matrix so that U.transpose()@M@U is skew-symmetric anti-diagonal with respect 
        to (n//2 x n//2) block matrices (Hereby M denotes the input matrix).
    '''
    assert all([abs((M.transpose() + M)[j, k]) < tol for j in range(len(M)) for k in range(len(M))]), f'Matrix not anti-symmetric within given tolerance {tol}.'
    U = youla_normal_form(M, **kwargs)
    My = U.transpose()@M@U
    U1 = _skew_post_youla(My)
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


def cortho_diagonalize_symmetric(M, tol=1e-14, **kwargs):
    '''
    Complex orthogonalize a complex symmetric matrix according to Thm. 4.4.27 in Horn & Johnson: Matrix Analysis 2nd Ed.
    This routine will compute a complex orthogonal matrix Y so that Y.transpose()@M@Y
    is diagonal with complex entries (where M denotes the complex symmetric input matrix). This means that
    that Y.transpose()@Y = 1 holds.
    
    Parameters
    ----------
    M:
        Complex symmetric matrix M to be diagonalized.
    
    Returns
    -------
    Y:
        Complex orthogonal matrix so that Y.transpose()@M@Y is diagonal with complex entries.
    '''
    assert all([abs((M.transpose() - M)[j, k]) < tol for j in range(len(M)) for k in range(len(M))]), f'Matrix not symmetric within given tolerance {tol}.'
    EV, ES = eigenspaces(M, tol=tol, **kwargs) # orthogonalization not required here; we just use the routine to get the eigenspaces for every eigenvalue
    Y = np.zeros([len(M), len(M)], dtype=complex)
    k = 0
    for subspace in ES:
        multiplicity = len(subspace)
        # Obtain Y-matrix as described in Horn & Johnson: Matrix Analysis 2nd. Edition, Lemma 4.4.26.
        X = np.array(subspace).transpose()
        U_sc = unitary_diagonalize_symmetric(X.transpose()@X)
        D = U_sc.transpose()@X.transpose()@X@U_sc
        Ri = np.diag([1/np.sqrt(D[k, k]) for k in range(len(D))])
        Y[:, k: k + multiplicity] = X@U_sc@Ri # this entry satisfies Y.transpose()@Y = 1
        k += multiplicity
    return Y


def cortho_symmetric_decomposition(M):
    '''
    If M is a complex non-singular square matrix, this routine will attempt to determine a decomposition 
        M = G@Q
    so that G is complex symmetric and Q is complex orthogonal.
    See Thm. 6.4.16 in Horn & Johnson: Topics in Matrix Analysis (1991).
    '''
    G = get_principal_sqrt(M@M.transpose())
    Q = np.linalg.inv(G)@M
    # checks:
    # G - G.transpose() == 0
    # Q.transpose()@Q = 1
    # G@Q = M
    return Q, G


def _diagonal2block(D, tol=1e-10):
    r'''
    Computes a unitary map U which will congruent-diagonalize a matrix D of the form
    
       D = diag(a, b, ..., -a, ..., -b, ...)
       
    to a block matrix B = U.transpose().conjugate()@D@U of the form
    
        /  0   W  \
    B = |         |
        \ -W   0  /
    
    where W = diag(a, b, ...).
    
    Parameters
    ----------
    D:
        A list of diagonal entries
        
    tol: float, optional
        A small parameter to identify the pairs on the diagonal of D (default 1e-10).
    
    Returns
    -------
    U:
        The unitary matrix described above.
    '''
    dim2 = len(D)
    assert dim2%2 == 0
    dim = dim2//2

    # Step 1: Determine the pairs on the diagonal which should be mapped.
    ind1, ind2 = [], []
    for i in range(len(D)):
        if i in ind1 or i in ind2:
            continue
        for j in range(len(D)):
            if j in ind1 or j in ind2 or j == i:
                continue
            if abs(D[i] + D[j]) < tol:
                ind1.append(i)
                ind2.append(j)
                break # index i consumed
                
    assert len(ind1) == len(ind2) and len(ind1) == dim, 'Error identifying pairs. Check input matrix or tolerance.'
    pairs = list(zip(ind1, ind2))
    
    # Step 2: Construct U, assuming that it will transform a matrix with the above order
    U = np.eye(dim2, dtype=complex)
    U2by2 = _create_umat_xieta(2)
    for i, j in pairs:
        U[i, i] = U2by2[0, 0]
        U[i, j] = U2by2[0, 1]
        U[j, i] = U2by2[1, 0]
        U[j, j] = U2by2[1, 1]
        
    # Step 3: Ensure that U maps to the desired block ordering.
    # Pair k should be mapped to the indices (k, k + dim) and (k + dim, k). This can be done with a
    # transformation T as follows.
    T = np.zeros([dim2, dim2])
    k = 0
    for i, j in pairs:
        # map ei to ek and ej to e(k + dim)
        T[k, i] = 1
        T[k + dim, j] = 1
        k += 1
        
    return U@T


def unitary_williamson(M, tol=1e-14, **kwargs):
    r'''
    Transform a symmetric invertible diagonalizable matrix M (which may be complex) 
    to a complex diagonal form, by means of a 
    complex matrix S: 
       S.transpose()@M@S = D,
    where S is symplectic:
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
    J = column_matrix_2_code(create_J(dim), code='numpy') # the default block symplectic structure 
    A = M12i@J@M12i
    
    EV, ES = np.linalg.eig(A)
    ESi = np.linalg.inv(ES)
    DD = ESi@A@ES # DD will be diagonal with the Krein-pairs of eigenvalues described above on its main diagonal.
    U = _diagonal2block(DD.diagonal(), tol=tol) # The unitary matrix U will transform DD into block-diagonal form with diagonal entries on the anti-diagonal via U.transpose()@DD@U.
    
    # Step 3: The routine ._diagonal2block internally determines the ordering of the Krein pairs. Since we require one representant for each pair, we compute the block-anti-diagonal result:
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
    

def anti_diagonalize_real_skew(M, code='numpy', **kwargs):
    r'''Anti-diagonalize a real skew symmetric matrix A so that it will have the block-form
    
             /  0   X  \
        A =  |         |
             \ -X   0  /
    
    where X is a diagonal matrix with positive entries.
    
    Attention: No check is made if the given matrix has real coefficients or is skew symmetric.
    
    Parameters
    ----------
    M:
        list of vectors defining a real skew symmetric (n x n)-matrix.

    **kwargs 
        Additional arguments are passed to .checks.relative_eq routine. Warning: If chosing 'mpmath' as code and dps value is
        set too small (corresponding to a too rough precision), then the tolerance 'tol' may have to be increased as well.
    
    Returns
    -------
    matrix
        Orthogonal real matrix R so that R.transpose()@M@R has skew-symmetric antidiagonal form with respect 
        to (n//2 x n//2) block matrices (Hereby M denotes the input matrix).
    '''
    evalues, evectors = eigenspaces(M, flatten=True, code=code, **kwargs)
    n = len(evalues)
    
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32)
        sqrt2 = mp.sqrt(2)
    
    # now construct a real basis
    v_block1, v_block2 = [], []
    processed_indices = []
    for i in range(1, n):
        if i in processed_indices:
            continue
        for j in range(i):
            if j in processed_indices:
                continue
            # pic those pairs of eigenvalues which belong to the same 'plane':
            same_plane = relative_eq(evalues[i].imag, -evalues[j].imag, **kwargs)
            if not same_plane:
                continue
                
            processed_indices.append(i)
            processed_indices.append(j)
            
            # Select the index belonging to the eigenvalue with positive imaginary part
            # Attention: This sets the signature of the matrix Omega in Eq. (6) in Williamson decomposition.
            # We want that the top right entry is positive and the bottom left entry is negative. Therefore:
            if evalues[i].imag >= 0: # the imaginary part of the eigenvalue of aj + 1j*bi (see below) will be positive
                pos_index = i
            else: # change the role of i and j (orientation) to maintain that ai + 1j*bi (see below) is positive; evalues[j].imag > 0
                pos_index = j
            
            ai = [(evectors[pos_index][k] + evectors[pos_index][k].conjugate())/sqrt2 for k in range(n)] # the 1/sqrt2-factor comes from the fact that evectors[pos_index] has been normalized to 1.
            bi = [-1j*(evectors[pos_index][k] - evectors[pos_index][k].conjugate())/sqrt2 for k in range(n)]
            v_block1.append(ai)
            v_block2.append(bi)
    return column_matrix_2_code(v_block1 + v_block2, code=code)


def williamson(V, code='numpy', **kwargs):
    r'''Compute Williamson's decomposition of a symmetric positive definite real matrix,
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
        
    J = column_matrix_2_code(create_J(dim), code=code)
    skewmat = V12i@J@V12i
    A = anti_diagonalize_real_skew(skewmat, code=code, **kwargs)    
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


def _create_umat_xieta(dim, code='numpy', **kwargs):
    '''
    Create a unitary matrix, mapping (p, q)-coordinates to (xi, eta)-coordinates via
    xi = (q + 1j*p)/sqrt(2)
    eta = (q - 1j*p)/sqrt(2)
    '''
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32) 
        sqrt2 = mp.sqrt(2)
        
    assert dim%2 == 0
    U1, U2 = [], []
    dim_half = dim//2
    for k in range(dim_half):
        k2 = k + dim_half
        U1.append([0 if i != k and i != k2 else 1/sqrt2 for i in range(dim)])
        U2.append([0 if i != k and i != k2 else 1j/sqrt2 if i == k else -1j/sqrt2 if i == k2 else 0 for i in range(dim)])
    return column_matrix_2_code(U1 + U2, code=code)

    
def normal_form(H2, T=[], code='numpy', **kwargs):
    r'''
    Perform linear calculations to transform a given second-order Hamiltonian,
    expressed in canonical coordinates (q, p), to
    complex normal form coordinates xi, eta. Along the way, the symplectic linear map to
    real normal form is computed. The quantities xi and eta are defined as in my thesis,
    Chapter 1.
    
    Paramters
    ---------
    H2:
        Symmetric matrix.
        
    T: matrix, optional
        Orthogonal matrix to change the ordering of canoncial
        coordinates and momenta, given here by default as (q1, ..., qn, p1, ..., pn), 
        into a different order. I.e. T transforms the 
        (n x n) block matrix
        
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
        into a matrix J' by matrix congruence: J' = T.transpose()@J@T.
        
    code: str, optional
        The code in which the matrix calculations should be performed. Supported codes: 'numpy', 'mpmath'.
        If the input is given in form of a matrix, the user has to ensure the correct code is selected.
        Default: 'numpy'.
             
    Returns
    -------
    dict
        Dictionary containing various linear maps. The items of this dictionary are described in the following.
        
        S: The symplectic map diagonalizing H2 via S.transpose()@D@S = H2, where D is a diagonal matrix.
        Sinv: The inverse of S, i.e. the symplectic map to (real) normal form.
        H2: The input in matrix form.
        T: The (optional) matrix T described above.
        J: The (original) symplectic structure J' = T.transpose()@J@T within which the input Hamiltonian was formulated.
            Hereby J is the block-matrix from above.
        J2: The new symplectic structure for the (xi, eta)-coordinates.
        U: The unitary map from the S(p, q) = (u, v)-block coordinates to the (xi, eta)-coordinates.
        Uinv: The inverse of U.
        K: The linear map transforming (q, p) to (xi, eta)-coordinates. K is given by U*S*T.
        Kinv: The inverse of K. Hence, it will transform H2 to complex normal form via Kinv.transpose()*H2*Kinv.
        rnf: The 'real' normal form, by which we understand the diagonalization of H2 relative to the 
            symplectic matrix S. Note that S might be complex if the Hesse matrix of H2 is not positive definite.
        cnf: The 'complex' normal form, which is given as the representation of H2 in terms of the complex
            normalizing (xi, eta)-coordinates (the 'new' complex symplectic structure).
    ''' 
    dim = len(H2)
    assert dim%2 == 0, 'Dimension must be even.'
        
    # Perform symplectic diagonalization
    if len(T) != 0: # transform H2 to default block ordering before entering williamson routine; the results will later be transformed back. This is easier instead of keeping track of orders inside the subroutines.
        H2 = T@H2@T.transpose()
        
    J = column_matrix_2_code(create_J(dim//2), code=code)
        
    if is_positive_definite(H2, code=code):
        S, D = williamson(V=H2, code=code, **kwargs)
        # The first dim columns of S denote (new) canonical coordinates u, the last dim columns of S
        # denote (new) canonical momenta v. We now get the block-matrix U, transforming the block-vector (u, v) to
        # (xi, eta) (as e.g. defined in my thesis):
        U = _create_umat_xieta(dim=dim, code=code, **kwargs)
        Sinv = -J@S.transpose()@J
    else:
        assert code == 'numpy' # TODO: mpmath
        # apply new general routine in case H2 is not positive definite
        Sinv, D, U = unitary_williamson(M=H2, **kwargs) # U.conjugate()@Sinv.transpose()@G@Sinv@U.conjugate().transpose() will be in (xi, eta)-canonical form.
        S = -J@Sinv.transpose()@J
    # U is hermitian, therefore
    Uinv = U.transpose().conjugate()

    # N.B. (p, J*q) = (Sp, J*S*q) = (u, J*v) = (Uinv*U*u, J*Uinv*U*v) = (Uinv*xi, J*Uinv*eta). Thus:
    J2 = Uinv.transpose()@J@Uinv # the new symplectic structure with respect to the (xi, eta)-coordinates (holds also in the case len(T) != 0)

    K = U@S # K(p, q) = (xi, eta)
    Kinv = Sinv@Uinv  # this map will transform to the new (xi, eta)-coordinates via Kinv.transpose()*H2*Kinv

    if len(T) != 0: # transform results back to the requested (q, p)-ordering
        S = T.transpose()@S@T
        Sinv = T.transpose()@Sinv@T
        J = T.transpose()@J@T
        H2 = T.transpose()@H2@T
        K = K@T
        Kinv = T.transpose()@Kinv
    
    # assemble output
    out = {}
    out['S'] = S 
    out['Sinv'] = Sinv # this symplectic map will diagonalize H2 in its original 
    # (q, p)-coordinates via Sinv.transpose()*H2*Sinv. Sinv (and S) are symplectic wrt. J
    out['H2'] = H2 # the input matrix
    out['rnf'] = Sinv.transpose()@H2@Sinv # the diagonal matrix obtained as a result of the symplectic diagonalization of H2
    out['T'] = T
    out['J'] = J # the original symplectic structure
    out['J2'] = J2 # the new symplectic structure
    out['U'] = U # the unitary map from the S(p, q)=(u, v)-block coordinates to the (xi, eta)-coordinates
    out['Uinv'] = Uinv
    out['K'] = K # K(q, p) = (xi, eta)
    out['Kinv'] = Kinv
    out['cnf'] = Kinv.transpose()@H2@Kinv # the representation of H2 in (xi, eta)-coordinates
    
    return out
    

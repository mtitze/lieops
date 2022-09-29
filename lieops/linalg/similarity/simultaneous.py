import numpy as np
import warnings
from scipy.linalg import schur

def off(*A):
    '''
    Obtain the sum of the Frobenius norm of a series of matrices, without the
    diagonal.
    '''
    return sum([np.sum(np.abs(Ak - np.diag(Ak.diagonal()))**2) for Ak in A])

def G(i, j, *A, tol=0):
    '''
    Determine the (complex) Jacobi parameters c and s according to Ref. [1].
    
    References
    ----------
    [1]: A. Souloumiac: "Jacobi Angles for Simultaneous Diagonalization", 
         SIAM Journal on Matrix Analysis and Applications, (1996).
    '''
    assert all([Ak.shape == A[0].shape for Ak in A])
    assert A[0].shape[0] == A[0].shape[1]
    G = np.zeros([3, 3])
    for Ak in A:
        hA = np.array([Ak[i, i] - Ak[j, j], Ak[i, j] + Ak[j, i], (Ak[j, i] - Ak[i, j])*1j])
        G += np.outer(hA.conj(), hA).real
    # G is symmetric and real, therefore
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    # By the spectral theorem, the eigenvalues of G must be real, since G is real & symmetric.
    # Obtain the eigenvector associated to the largest eigenvalue of G
    x, y, z = eigenvectors.transpose()[np.argsort(eigenvalues)][-1] # x, y, z are real as well (see also the discussion regarding the eigh routine here: https://stackoverflow.com/questions/31572871/finding-the-real-eigenvectors-of-a-real-symmetric-matrix-in-numpy-or-scipy)
    # according to the docs in eigh, the eigenvectors are already normalized.
    c_abs = np.sqrt(np.abs((1 + x)/2)) # abs ensures that we do not run into negative values if x ~ -1
    s_abs = np.sqrt(np.abs((1 - x)/2)) # abs ensures that we  ... if x ~ 1
    if s_abs == 0:
        s = 0
        c = 1 # |c| = 1, phi = 0 (see below), therefore c = 1 here.
    elif c_abs == 0:
        c = 0
        s = 1 # |c| = 0, phi = 0 (see below). Then s must be +1/-1. We chose s = 1. 
    else:
        exp_i_phi_psi = (y - z*1j)/2/s_abs/c_abs # if c = |c|*exp(i phi) and s = |s|*exp(i psi), then 
        # exp_i_phi_psi = exp(i*(phi + psi)).
        # since c and s are determined up to a phase, we set phi = 0.
        c = c_abs
        s = s_abs*exp_i_phi_psi
    return c, s

def givens_rotation(i, j, c, s, n):
    '''
    Return a (unitary) Givens-rotation in the plane (i, j), with respect to complex quantities
    c and s in dimension n.
    '''
    result = np.eye(n, dtype=np.complex128)
    result[i, i] = c
    result[i, j] = s.conjugate()
    result[j, i] = -s
    result[j, j] = c.conjugate()
    return result

def simuldiag(*A, tol: float=1e-14, max_sweeps: int=100):
    '''
    Simultaneously diagonalize a given set of normal matrices. 
    
    Note that a matrix A is said to be
    *normal*, iff 
    A@A^H = A^H@A 
    holds, where ^H is complex conjugation and transposition.
    
    Parameters
    ----------
    A: ndarray(s)
        A series of normal matrices to be simultaneously diagonalized.
    
    tol: float, optional
        A tolerance to determine when the algorithm should stop. This tolerance is connected
        to the Frobenius norm of the sum of the off-diagonal parts of the matrices.
        
    max_sweeps: int, optional
        An integer to determine the maximal number of 'sweeps' (Givens-rotations)
        the algorithm will perform if not converging.
        
    Returns
    -------
    Q: ndarray
        A unitary matrix diagonalizing the given matrices, 
        so that Q@A@Q^H is diagonal, for every matrix A.
        
    A: ndarray(s)
        The diagonalized matrices.
        
    err: float
        The difference of the sum of Frobenius norms of the off-diagonal parts of the diagonal
        matrices with respect to the original matrices (see the source code / Ref. [2]).
        
    n_sweeps: int
        The number of sweeps used in the algorithm.
        
    References
    ----------
    [1]: A. Souloumiac: "Jacobi Angles for Simultaneous Diagonalization", 
         SIAM Journal on Matrix Analysis and Applications, (1996).
         
    [2]: A. Bunse-Gerstner and V. Mehrmann: "Numerical Methods for Simultaneous Diagonalization",
         SIAM Journal on Matrix Analysis and Applications, (1998).
    '''
    AA = A[0]
    assert all([Ak.shape == AA.shape for Ak in A])
    assert AA.shape[0] == AA.shape[1]
    n = AA.shape[0]
    
    # Prepare matrices by applying Schur's diagonalization of the first matrix to all others.
    # see comment in Ref. [2]:
    DD, Q = schur(AA, output='complex')
    Q = Q.transpose().conj() # to be in line with the code inside the loop below
    A = [Q@Ak@Q.transpose().conj() for Ak in A]
    
    # Now apply the actual algorithm.
    lower_bound = tol*sum([np.linalg.norm(Ak) for Ak in A]) # The bound in Algorithm 1 in Ref. [2]; it is invariant under unitary transformations, so that we can safely compute this quantity outside of the while loop
    n_sweeps = 1
    while off(*A) > lower_bound:
        for i in range(n):
            for j in range(i + 1, n):
                c, s = G(i, j, *A)
                R = givens_rotation(i, j, c, s, n) 
                # TODO: Improvement: if (i, j) and (k, l) are disjoint, we may perform R simultaneously
                Q = R@Q
                A = [R@Ak@R.transpose().conj() for Ak in A]
                
        n_sweeps += 1
        if n_sweeps > max_sweeps:
            warnings.warn(f'Simultaneous diagonalization does not converge within {max_sweeps} sweeps, using a tolerance of {tol}.')
            break
    return Q, A, abs(off(*A) - lower_bound), n_sweeps
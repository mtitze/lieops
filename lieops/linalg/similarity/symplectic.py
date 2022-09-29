import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

from lieops.linalg.matrix import create_J
from lieops.linalg.misc import identifyPairs
from lieops.linalg.similarity.simultaneous import simuldiag

'''
The notation of the routines follows the description in [1].

Reference(s):
[1] R. de la Cruz and H. Fassbender: "On the diagonalizability of a matrix by a symplectic equivalence, 
    similarity or congruence transformation", Linear Algebra and its Applications 496 (2016) pp. 288 -- 306.
'''

def lemma9(u):
    u = np.array(u, dtype=np.complex128)
    norm_u = np.linalg.norm(u)
    assert norm_u > 0
    v = u/norm_u # |v| = 1
    # construct unitary matrix U so that U(e_1) = v; for the return matrix U it holds:
    # 0) det(U) = |v[0]|**2 + |v[1]|**2 = 1
    # 1) U@e_1 = a*u for a real number a (U@e_1 = v = u/norm_u and 1/norm_u is real)
    # 2) 1 = U.conj().transpose()@U
    # 3) J = U.transpose()@J@U, since det U = 1 and SL(2; C) = Sp(2; C) in this case
    return np.array([[v[0], -v[1].conj()], [v[1], v[0].conj()]]) # the second column of P contains a vector orthogonal to v


def thm10(u, tol=0):
    '''
    For every complex vector u construct a unitary and symplectic matrix U so that U(e_1) = u/|u| holds.
    
    Parameters
    ----------
    u: array-like
        A complex 2n-dimensional vector.
        
    Returns
    -------
    U: array-like
        A complex (2n)x(2n)-dimensional unitary and symplectic matrix.
    '''
    u = np.array(u, dtype=np.complex128)
    norm_u = np.linalg.norm(u)
    assert norm_u > 0
    v = u/norm_u
    dim2 = len(u)
    assert dim2%2 == 0
    dim = dim2//2
    P = np.zeros([dim2, dim2], dtype=np.complex128)
    for k in range(dim):
        vpart = [v[k], v[k + dim]]
        if np.linalg.norm(vpart) != 0: # TODO: may use tol instead.
            Ppart_inv = lemma9(vpart) # det(Ppart_inv) = 1
            Ppart = np.array([[Ppart_inv[1, 1], -Ppart_inv[0, 1]], [-Ppart_inv[1, 0], Ppart_inv[0, 0]]])
        else:
            Ppart = np.eye(2)
        P[k, k] = Ppart[0, 0]
        P[k, k + dim] = Ppart[0, 1]
        P[k + dim, k] = Ppart[1, 0]
        P[k + dim, k + dim] = Ppart[1, 1]
    
    w_full = P@v # N.B. w_full is real: If v[k, k + dim] has norm != 0, then, by construction of lemma9 routine, the scaling
    # factors induced by P on the e_k-vectors are real. If v[k, k + dim] has norm 0, then both its components are zero and
    # therefore also the result. In any case w_full is real.
    
    # Furthermore we have |w_full| = |P@v| = |v| = 1

    if tol > 0:
        # optional consistency checks
        assert abs(np.linalg.norm(w_full) - 1) < tol
        assert all([abs(w_full[k + dim]) < tol for k in range(dim)]) # if this fails, then something is definititvely wrong in the code and needs to be investigated. The components from dim to 2*dim must be zero, because by construction the map P consists of individual 2x2-maps, each mapping into their e_1-component (thus the second component is always zero).
        assert all([abs(w_full[k].imag) < tol for k in range(dim2)]) # if this fails, this is basically not a problem, it just checks the above considerations on w_full. But it may indicate a hidden error in our line of thought and should be investigated as well. See also the comments inside 'lemma9'-routine.
    
    # Compute a Householder matrix HH so that HH@w = e_1 holds.
    w = w_full[:dim]
    diff = w.copy().reshape(dim, 1) # without .copy(), the changes on 'diff' below would lead to changes in w; reshaping to be able to compute the dyadic product below
    diff[0, 0] -= 1
    norm_diff = np.linalg.norm(diff)
    if norm_diff > 0:
        diff = diff/norm_diff # so that diff = (w - e_1)/|w - e_1|
        HH = np.eye(dim) - 2*diff@diff.transpose().conj() # N.B. diff is real by the above considerations
    else:
        # w == e_1
        HH = np.eye(dim)
    # Using the Householder matrix, construct V, as given in Thm. 10
    zeros = np.zeros(HH.shape)
    V = np.block([[HH, zeros], [zeros, HH.conj()]]) # N.B. V is real by the above considerations
    # now it holds V@P@v = V@w_full = HH@w = e_1 and V@P is unitary and symplectic.
    return (V@P).transpose().conj()


def cor29(A, **kwargs):
    r'''
    Let A be a normal and (skew)-Hamiltonian. Recall that this means A satisfies the following two conditions:
    
    1) A.transpose()@J + sign*J@A = 0, where sign = 1 if "skew", else -1.
    2) A.transpose().conj()@A = A@A.transpose().conj()
    
    Then this routine will find a symplectic and unitary matrix U so that
    
    U.transpose().conj()@A@U
    
    is diagonal. It is hereby assumed that J correspond to the symplectic structure in terms of nxn-block matrices:
    
        / 0   1 \
    J = |       |
        \ -1  0 /
    
    Parameters
    ----------
    A: array-like
        A complex-valued (2n)x(2n)-matrix having the above properties (Warning: No check is made against these properties within
        this routine).
        
    Returns
    -------
    U: array-like
        A complex-valued symplectic and unitary (2n)x(2n)-matrix U so that U^(-1)@A@U = D is diagonal.
    '''
    A = np.array(A, dtype=np.complex128)
    assert A.shape[0] == A.shape[1]
    dim2 = A.shape[0]
    assert dim2%2 == 0
    dim = dim2//2
    
    # get one eigenvalue and corresponding eigenvector of the given matrix
    if dim2 > 2:
        eigenvalues, eigenvectors = eigs(A, k=1)
    else: # scipy.sparse.linalg.eigs will throw a RuntimeWarning in the case k=1, dim2 = 2. 
          # To prevent these warnings, we compute an eigenvalue and eigenvector of the 2x2 matrix A with scipy.linalg.eig here.
        eigenvalues, eigenvectors = eig(A)        
    eigenvector = eigenvectors[:, 0]
    v = eigenvector/np.linalg.norm(eigenvector)
    U = thm10(v, **kwargs) # So that U(e_1) = v holds.
    U_inv = U.transpose().conj()
    B = U_inv@A@U
    
    if dim >= 2:
        # obtain submatrix from B; TODO: May use masked array, if code is stable
        B11 = B[1:dim, 1:dim]
        B12 = B[1:dim, dim + 1:]
        B21 = B[dim + 1:, 1:dim]
        B22 = B[dim + 1:, dim + 1:]
        U_sub = cor29(np.block([[B11, B12], [B21, B22]]), **kwargs)
        
        dim_sub = dim - 1
        # combine U with U_sub;
        # 1) Split U_sub at its dimension in half
        U_sub_11 = U_sub[:dim_sub, :dim_sub]
        U_sub_12 = U_sub[:dim_sub, dim_sub:]
        U_sub_21 = U_sub[dim_sub:, :dim_sub]
        U_sub_22 = U_sub[dim_sub:, dim_sub:]
        
        # 2) Include the individual U_sub components into the grander U-matrix
        zeros_v = np.zeros([1, dim_sub])
        zeros_vt = np.zeros([dim_sub, 1])
        one = np.array([[1]])
        zero = np.array([[0]])
        
        U_sub_full = np.block([[one,      zeros_v, zero,     zeros_v],
                               [zeros_vt, U_sub_11, zeros_vt, U_sub_12],
                               [zero,     zeros_v, one,      zeros_v],
                               [zeros_vt, U_sub_21, zeros_vt, U_sub_22]])
        U = U@U_sub_full
    
    return U


def thm31(M, tol1=1e-14, tol2=0, **kwargs):
    r'''
    Find a unitary and symplectic matrix U which can diagonalize a given complex normal and J-normal matrix M,
    so that U@M@U^(-1) is diagonal.
    
    Note that the above properties of M translate to:
    1) M@M^H = M^H@M (M normal),
    2) M@J@M.transpose()@J = J@M.transpose()@J@M (M J-normal).
    
    In particular, if M is symplectic, M will be J-normal.
    
    Hereby J denotes the (2n)x(2n)-matrix describing the standard symplectic structure
    in block-matrix form:
    
        /  0  1 \
    J = |       |
        \ -1  0 /
    
    U symplectic and unitary means:
    1) U@U^H = 1 (= U^H@U) (U unitary)
    2) U.transpose()@J@U = J (U symplectic)
    
    The algorithm follows the one outlined in Ref. [1].
    
    Parameters
    ----------
    M: ndarray
        The matrix M, having the above properties, which should be diagonalized.
        Attention: No extensive checks are made against the properties M has to satisfy.
        
    tol1: float, optional
        An optional tolerance to identify certain pairs of eigenvalues (see source).

    tol2: float, optional
        An optional tolerance to perform some consistency checks on the output, if > 0.

    sdn_tol: float, optional
        The 'tol' parameter for the lieops.linalg.similarity.simultaneous.simuldiag routine.
    
    Returns
    -------
    T: ndarray
        A complex unitary and symplectic matrix as described above.
    
    References
    ----------
    [1] R. de la Cruz and H. Fassbender: "On the diagonalizability of a matrix by a symplectic equivalence, 
        similarity or congruence transformation", Linear Algebra and its Applications 496 (2016) pp. 288 -- 306. 
    '''
    assert M.shape[0] == M.shape[1]
    dim2, _ = M.shape
    assert dim2%2 == 0
    dim = dim2//2
    
    # Obtain the unitary and symplectic matrix P which diagonalizes M - phi_J(M) as in Thm. 31.
    # Hereby MphiJM := M - phi_J(M) = M + J@M.transpose()@J is skew-J-symmetric, i.e. 
    # -J@MphiJM.transpose()@J = phi_J(MphiJM) = -MphiJM, or
    # MphiJM.transpose()@J + J@MphiJM = 0, which means that
    # MphiJM is a Hamiltonian matrix.
    # It follows that if x is an eigenvalues of MphiJM, then also -x is an eigenvalue. 
    # In the following context we are interested in identifying those pairs of eigenvalues.
    J = create_J(dim)
    MphiJM = M + J@M.transpose()@J
    Pi = cor29(MphiJM) # then D = P@MphiM@P^(-1) is diagonal; P is unitary and symplectic.
    P = Pi.transpose().conj()
    D = P@MphiJM@Pi
    
    # Determine the non-zero pairs on the diagonal:
    diag = D.diagonal()
    pairs = identifyPairs(diag, condition=lambda a, b: abs(a + b) < tol1 and abs(a) > tol1)
    PMPi = P@M@Pi
    
    # Construct unitary and symplectic matrix T:
    T = np.zeros([dim2, dim2], dtype=np.complex128)

    if len(pairs) > 0: # N.B. it may happen that len(pairs) == 0 in which case M1 and M2 (see below) can not be created (e.g. if M == 1).
        M1_indices, M2_indices = zip(*pairs)
        M1 = np.array([[PMPi[i, j] for j in M1_indices] for i in M1_indices])
        M2 = np.array([[PMPi[i, j] for j in M2_indices] for i in M2_indices])
                
        # Simultaneously diagonalize M1 and M2.transpose() by a unitary transformation X
        X, D1D2, err, n_sweeps = simuldiag(M1, M2.transpose(), tol=kwargs.get('sdn_tol', 1e-14))

        # Construct unitary and symplectic matrix Q diagonalizting M1 oplus M2:
        zerob = np.zeros(X.shape, dtype=np.complex128)
        Q = np.block([[X, zerob], [zerob, X.conj()]])
    
        Q_indices = M1_indices + M2_indices
        for k in range(len(Q_indices)):
            for l in range(len(Q_indices)):
                kk = Q_indices[k]
                ll = Q_indices[l]
                T[kk, ll] = Q[k, l]
                
        if tol2 > 0: # consistency checks
            dim1 = len(M1_indices) # len(M1_indices) = len(M2_indices) by construction with zip
            J1 = create_J(dim1)
            zero1 = M1@M2.transpose() - M2.transpose()@M1
            zero2 = M1.transpose().conj()@M1 - M1@M1.transpose().conj()
            zero3 = M2.transpose().conj()@M2 - M2@M2.transpose().conj()
            zero4 = Q.transpose().conj()@Q - np.eye(dim1*2) # Q must be unitary
            zero5 = Q.transpose()@J1@Q - J1 # Q must be symplectic
            for zero in [zero1, zero2, zero3, zero4, zero5]:
                assert all([abs(zero[i, j]) < tol2 for i in range(dim1) for j in range(dim1)])
    
    M3_indices = np.where(np.abs(diag) < tol1)[0]    
    if len(M3_indices) > 0:
        M3 = np.array([[PMPi[i, j] for j in M3_indices] for i in M3_indices])
        P3i = cor29(M3)
        P3 = P3i.transpose().conj()
        for k in range(len(M3_indices)):
            for l in range(len(M3_indices)):
                kk = M3_indices[k]
                ll = M3_indices[l]
                T[kk, ll] = P3[k, l]
    T = T@P
                    
    if tol2 > 0: # consistency checks
        zero6 = T.transpose().conj()@T - np.eye(dim2) # T must be unitary
        zero7 = T.transpose()@J@T - J # T must be symplectic
        T_inv = -J@T.transpose()@J
        TMTi = T@M@T_inv
        zero8 = TMTi - np.diag(TMTi.diagonal()) # T@M@T_inv must be diagonal
        for zero in [zero6, zero7, zero8]:
            a, b = zero.shape
            assert all([abs(zero[i, j]) < tol2 for i in range(a) for j in range(b)])
            
    return T
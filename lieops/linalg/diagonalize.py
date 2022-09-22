import numpy as np
from scipy.sparse.linalg import eigs

'''
Implementation of various diagonalization routines as given in Ref. [1]. 
The notation of the routines follows the description in [1].

Reference(s):
[1] R. de la Cruz and H. Fassbender: "On the diagonalizability of a matrix by a symplectic equivalence, similarity or congruence transformation", Linear Algebra and its Applications 496 (2016) pp. 288 -- 306.
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
    For every complex vector u construct a unitary and symplectic matrix U so that U(e_1) = a*u holds, where
    'a' is a complex number.
    
    Parameters
    ----------
    u: array-like
        A complex 2n-dimensional vector.
        
    Returns
    -------
    U: array-like
        A complex (2n)x(2n)-dimensional unitary and symplectic matrix so that U(e_1) = a*u holds, where 'e_1' is
        the first unit vector and 'a' is a complex number.
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

    if tol > 0:
        # optional consistency checks
        assert all([abs(w_full[k + dim]) < tol for k in range(dim)]) # if this fails, then something is definititvely wrong in the code and needs to be investigated. The components from dim to 2*dim must be zero, because by construction the map P consists of individual 2x2-maps, each mapping into their e_1-component (thus the second component is always zero).
        assert all([abs(w_full[k].imag) < tol for k in range(dim2)]) # if this fails, this is basically not a problem, it just checks the above considerations on w_full. But it may indicate a hidden error in our line of thought and should be investigated as well. See also the comments inside 'lemma9'-routine.
    
    # Compute a Householder matrix HH so that HH@w = e_1 holds. The second equation holds because |w| = 1.
    w = w_full[:dim]
    diff = w.copy().reshape(dim, 1) # without .copy(), the changes on 'diff' below would lead to changes in w; reshaping to be able to compute the dyadic product below
    diff[0, 0] -= 1
    norm_diff = np.linalg.norm(diff)
    if norm_diff > 0:
        diff = diff/norm_diff # so that diff = (w - e_1)/|w - e_1|
        HH = np.eye(dim) - 2*diff@diff.transpose().conj()
    else:
        # w == e_1
        HH = np.eye(dim)
    # Using the Householder matrix, construct V, as given in Thm. 10
    zeros = np.zeros(HH.shape)
    V = np.block([[HH, zeros], [zeros, HH.conj()]])
    # now it holds V@P@v = V@w_full = HH@w = e_1 and V@P is unitary and symplectic.
    return (V@P).transpose().conj()


def cor29(A):
    r'''
    Let A be a normal and (skew)-Hamiltonian. Recall that this means A satisfies the following two conditions:
    
    1) A.transpose()@J + sign*J@A = 0, where sign = 1 if "skew", else -1.
    2) A.transpose().conj()@A = 1
    
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
    eigenvalues, eigenvectors = eigs(A, k=1)
    eigenvector = eigenvectors[:,0]
    v = eigenvector/np.linalg.norm(eigenvector)
    U = thm10(v)
    U_inv = U.transpose().conj()
    B = U_inv@A@U
    
    if dim >= 2:
        # obtain submatrix from B; TODO: May use masked array, if code is stable
        B11 = B[1:dim, 1:dim]
        B12 = B[1:dim, dim + 1:]
        B21 = B[dim + 1:, 1:dim]
        B22 = B[dim + 1:, dim + 1:]
        U_sub = cor29(np.block([[B11, B12], [B21, B22]]))
        
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


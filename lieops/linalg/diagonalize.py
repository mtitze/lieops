import numpy as np

'''
Implementation of various diagonalization routines. The notation follows the paper [1].

Reference(s):
[1] R. de la Cruz and H. Fassbender: "On the diagonalizability of a matrix by a symplectic equivalence, similarity or congruence transformation", Linear Algebra and its Applications 496 (2016) pp. 288 -- 306
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
    u = np.array(u, dtype=np.complex128)
    norm_u = np.linalg.norm(u)
    assert norm_u > 0
    v = u/norm_u
    dim2 = len(u)
    dim = dim2//2
    P = np.zeros([dim2, dim2], dtype=np.complex128)
    for k in range(dim):
        vpart = [v[k], v[k + dim]]
        if np.linalg.norm(vpart) != 0:
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
    return V@P
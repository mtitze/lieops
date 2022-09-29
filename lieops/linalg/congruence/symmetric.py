import numpy as np
from lieops.linalg.common import eigenspaces
from .youla import unitary_diagonalize_symmetric

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


import numpy as np
import pytest

from lieops.linalg.matrix import create_J
from lieops.linalg.similarity.symplectic import cor29, thm31
from lieops.linalg.misc import identifyPairs

from scipy.linalg import expm

#np.random.seed(123456789) # for some seeds the tests below may fail; requires investigation
np.random.seed(1234785987) # for some seeds the tests below may fail; requires investigation

def create_spn_matrix(dim, max_path=2*np.pi, tol=0):
    r'''
    Create an element of Sp(n) = Sp(2n; C) \cap U(2n), the compact and simply connected group of
    symplectic and unitary matrices.
    
    Parameters
    ----------
    dim: int
        Controls the dimension of the requested matrix. A matrix of shape
        (2*dim)x(2*dim) will be returned.
    
    max_path: float, optional
        A scaling factor to traverse the lie-algebra.
        
    tol: float, optional
        An optional tolerance to perform basic checks, if a value > 0 is provided.
        
    Returns
    -------
    S: array-like
        The respective element in the Lie-group Sp(2n).
        
    X: array-like
        A (2*dim)x(2*dim) matrix within the Lie-algebra of Sp(2n). It holds exp(X) = S.
    '''
    # Step 1: Construct a random unitary symplectic matrix
    P = (1 - 2*np.random.rand(dim, dim)) + (1 - 2*np.random.rand(dim, dim))*1j
    P = P + P.transpose()
    Q = (1 - 2*np.random.rand(dim, dim)) + (1 - 2*np.random.rand(dim, dim))*1j
    Q = Q + -Q.transpose().conj()
    X = np.block([[Q, P], [-P.conj(), Q.conj()]])*max_path
    
    # now use the fact that exp is surjective onto Sp(n):
    S = expm(X)

    if tol > 0:
        dim2 = 2*dim
        J = create_J(dim)
        zero1 = X.transpose()@J + J@X # X in sp(2n; C)
        zero2 = X.transpose().conj() + X # X in u(2n)
        zero3 = S.transpose()@J@S - J # S in Sp(2n; C)
        zero4 = S.transpose().conj()@S - np.eye(dim2) # S in U(2n)
        for zero in [zero1, zero2, zero3, zero4]:
            assert all([abs(zero[i, j]) < tol for i in range(dim2) for j in range(dim2)])  
    
    return S, X


#########
# Tests #
#########

@pytest.mark.parametrize("dim, skew", [(2, 1), (2, 1), (4, 1), (4, 1), (6, 1), (6, 1)] + 
                         [(2, -1), (2, -1), (4, -1), (4, -1), (6, -1), (6, -1)])
def test_cor29(dim, skew, tol=1e-14, **kwargs):
    '''
    Test Corollary 29 for the special normal and (skew)-Hamiltonian (J-(skew)-symmetric) matrices
    M +- J@M.transpose()@J ,
    where M is an element in Sp(n) (i.e. a special normal and J-normal matrix.)
    
    Parameters
    ----------
    dim: int
        The dimension of the problem
        
    skew: int
        Defines whether to check for skew (1) or non-skew (-1) matrices.

    tol: float, optional
        A tolerance to check values against zero    
    '''
    S, _ = create_spn_matrix(dim, tol=tol, **kwargs)
    
    # Using the unitary and symplectic matrix S, construct a (particular) 
    # normal and (skew)-Hamiltonian matrix W:
    J = create_J(dim)
    W = S + J@S.transpose()@J*skew # = M - phi_J(M)*sign in Ref. [1]
    zero1 = W.conj().transpose()@W - W@W.conj().transpose() # phiJS is normal
    zero2 = W.transpose()@J + J@W*skew # phiJS is (skew)-Hamiltonian
    
    # Apply Cor. 29 on W:
    U = cor29(W, tol=tol)
    
    zero3 = U.transpose().conj()@U - np.eye(dim*2)
    zero4 = U.transpose()@J@U - J
    for zero in [zero1, zero2, zero3, zero4]:
        assert all([abs(zero[i, j]) < tol for i in range(dim*2) for j in range(dim*2)])
    
    D = U.transpose().conj()@W@U
    for j in range(dim*2):
        # check if the off-diagonal entries are all zero
        assert all([abs(D[i, j]) < tol and abs(D[j, i]) < tol for i in range(j)])
    diag = D.diagonal()
    n_diag = [d for d in diag if abs(d) >= tol] # the non-zero eigenvalues of D
    pairs = identifyPairs(n_diag, condition=lambda a, b: abs(a - b.conj()) < tol)
    assert len(n_diag)%2 == 0
    n_diag_dim = len(n_diag)//2
    for i, j in pairs:
        # by construction in Cor. 29, the pairs of eigenvalues should be sorted with respect to (dim, 2*dim)-block-structure.
        # If this last check fails, then something odd could have happened at the representation of U^(-1)@A@U in the proof of Lemma 28 in Ref. [1]: 
        # In our scripts we always assumed that the matrices A_j for j = 1, 2, 3, 4 have the same shape (namely they are (n - 1)x(n - 1)-matrices).
        assert j == i + n_diag_dim


@pytest.mark.parametrize("dim, tol", [(1, 1e-13), (1, 1e-13), (2, 1e-13), (2, 1e-13), (3, 1e-12), (3, 1e-12), (4, 5e-12), (4, 5e-12)])
def test_thm31(dim, tol, tol1=1e-13):
    '''
    Test Theorem 31 (thm31 routine) for the specific case of Sp(n)-matrices.
    '''
    S, _ = create_spn_matrix(dim)
    _ = thm31(S, tol1=tol1, tol2=tol) # checks are done within thm31 if a tolerance is provided.
    
    
inp = []
for dim1, dim2, tol in [(1, 1, 1e-12), (1, 2, 5e-12), (2, 2, 5e-12), (2, 3, 5e-12)]:
    T1, _ = create_spn_matrix(dim1)
    T2, _ = create_spn_matrix(dim2)
    inp.append((T1, T2, tol))
    inp.append((np.eye(dim1*2), T2, tol))
       
@pytest.mark.parametrize("T1, T2, tol", inp)
def test_thm31_2(T1, T2, tol, tol1=1e-12, sdn_tol=1e-12):
    '''
    Test Theorem 31 if combining two matrices into a grander Sp(n)-matrix.
    '''

    
    # create a (special) grand symplectic and unitary matrix
    a2, b2 = T1.shape
    c2, d2 = T2.shape

    assert a2 == b2 and c2 == d2
    assert a2%2 == 0
    assert c2%2 == 0
    assert a2 + c2 == b2 + d2
    dim2 = a2 + c2
    dim = dim2//2 # dim = a + c
    a, c = a2//2, c2//2

    mixed = np.zeros([dim2, dim2], dtype=np.complex128)

    for i in range(a):
        for j in range(a):
            mixed[i, j] = T1[i, j]
            mixed[i, j + dim] = T1[i, j + a]
            mixed[i + dim, j] = T1[i + a, j]
            mixed[i + dim, j + dim] = T1[i + a, j + a]

    for i in range(c):
        for j in range(c):
            ia = i + a
            ja = j + a

            mixed[ia, ja] = T2[i, j]
            mixed[ia, ja + dim] = T2[i, j + c]
            mixed[ia + dim, ja] = T2[i + c, j]
            mixed[ia + dim, ja + dim] = T2[i + c, j + c]
            
    _ = thm31(mixed, tol1=tol1, tol2=tol, sdn_tol=sdn_tol)

    

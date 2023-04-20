import numpy as np

def LDL_cholesky1(A):
    '''
    Compute the LDL-Cholesky decomposition of a Hermitian positive definite matrix A
    so that it holds
    A == L@D@L.transpose().conjugate()
    where L is a (in general complex) lower triangular matrix with units on its diagonal.
    
    Parameters
    ----------
    A: ndarray
        Matrix to be factorized.
        
    Returns
    -------
    L: ndarray
    D: ndarray
    Di12: ndarray
    '''
    C = np.linalg.cholesky(A) # C@C^* == A with C lower trianglular
    Cd = C.diagonal()
    Si = np.diag(Cd**-1)
    L = C@Si # thus C = L@S, so that L@S@S^*@L^* = A. We thus have D = S@S^*:
    D = np.diag(Cd*Cd.conjugate())
    # Moreover, the diagonal entries of L are 1, by construction.
    return L, D

def iwasawa(X):
    '''
    Compute matrices A, N in the Iwasawa decomposition 
    X = K@A@N
    of a complex symplectic matrix X according to Ref. [1].
    
    Parameters
    ----------
    X: ndarray
        Matrix to be factorized.
        
    Returns
    -------
    A: ndarray
    N: ndarray
    
    Reference(s)
    ------------
    [1]: T.-Y. Tam: "Computing the Iwasawa decomposition of a symplectic matrix by 
    Cholesky factorization", Applied Mathematics Letters 19 (2006), p. 1421 – 1424.
    '''
    dim2 = len(X)
    assert dim2%2 == 0, 'Dimension not even.'
    dim = dim2//2
    XX = X.transpose().conjugate()@X
    A1, B1 = XX[:dim,:dim], XX[:dim, dim:]
    Qs, H = LDL_cholesky1(A1) # A1 == Qs@H@Qs.transpose().conjugate()
    Hi = np.diag(H.diagonal()**-1)
    Q = Qs.transpose().conjugate()
    A = np.zeros([dim2, dim2], dtype=np.complex128)
    H12_d = np.sqrt(H.diagonal())
    H12 = np.diag(H12_d)
    Hi12 = np.diag(H12_d**-1)
    A[:dim, :dim] = H12
    A[dim:, dim:] = Hi12
    Qi = np.linalg.inv(Q) # ... improvement possible here?
    Qitr = Qi.transpose()
    N = np.zeros([dim2, dim2], dtype=np.complex128)
    N[:dim, :dim] = Q
    N[:dim, dim:] = Hi@Qitr@B1
    N[dim:, dim:] = Qitr
    return A, N
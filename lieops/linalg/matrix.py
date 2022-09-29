# This file collects routines which are focused on fundamental generation of specific
# matrices and to perform basic matrix operations.

import numpy as np
import mpmath as mp

def printmat(M, tol=1e-14):
    # print a matrix (for e.g. debugging reasons)
    M = mp.matrix(M)
    mp.nprint(mp.chop(M, tol))

def create_J(dim: int):
    r'''
    Create a 2*dim-square matrix J, corresponding to the standard 
    symplectic block-matrix
    
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
    Parameters
    ----------
    dim: int
        Dimension/2 of the matrix to be constructed.
        
    Returns
    -------
    list
        List of column vectors.
    '''
    dim2 = 2*dim
    J1, J2 = [], []
    for k in range(dim):
        J1.append([0 if i != k + dim else -1 for i in range(dim2)])
        J2.append([0 if i != k else 1 for i in range(dim2)])
    return np.array(J1 + J2).transpose()

def expandingSum(pairs):
    '''Compute a transformation matrix T, to transform a given
    (2n)x(2n) matrix M, represented in (q1, p1, q2, p2, ..., qn, pn)-coordinates, into
    a (q1, q2, ..., qn, p1, p2, ..., pn)-representation via
    M' = T^(-1)@M@T. T will be orthogonal (and unitary), i.e. T^(-1) = T.transpose().
    
    See also Refs. [1, 2] or (alternatively) in Ref. [3], p. 292. In particular, M
    is given in terms of 2x2 block matrices, then M' is called the 'expanding Sum' of M.
    
    Parameters
    ----------
        
    pairs: int or list
        If an integer is given, then it is assumed that this integer denotes
        the dimension of the current problem. A respective square matrix T
        (array-like) will be constructed, as described above.
        
        If a list is given, then it is assumed that this list consists of
        tuples by which one can tweak the order of the original coordinates:
        In this case the list must unambigously identify each pair (q_i, p_j) by a specific tuple
        (i, j) in the list. The outcome will be an orthogonal (and unitary) matrix T
        which transformations the coordinates (q_i, ..., p_j, ...) into 
        (q_i, ..., q_n, p_j, ..., p_n). Herby the k'th element in the list will be cast to
        positions k, k + dim.
        
    Returns
    -------
    np.matrix
        Numpy matrix T defining the aforementioned transformation.
        
    Reference(s):
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonances for the CERN PS and SPS", PhD Thesis (2019)
    [2]: M. Titze: "On emittance and optics calculation from the tracking data in periodic lattices", arXiv.org (2019)
    [3]: R. J. de la Cruz and H. Faßbender: "On the diagonalizability of a matrix by a symplectic equivalence, similarity or congruence transformation (2016).
    '''
    if type(pairs) == int:
        dim = pairs
        # the 'default' ordering is used, transforming (q1, p1, q2, p2, ...) into (q1, q2, ..., p1, p2, ...)
        indices_1 = [(j, j//2) if j%2 == 0 else (j, dim + (j - 1)//2) for j in range(2*dim)]
    else:
        dim = len(pairs)
        indices_1 = [(pairs[k][0], k) for k in range(dim)] + [(pairs[k][1], k + dim) for k in range(dim)]
        
    T = np.zeros([dim*2, dim*2])
    # define the columns of T:
    for i, j in indices_1:
        T[i, j] = 1
    return T

def matrix_from_dict(M, symmetry: int=0, **kwargs):
    '''
    Create matrix from (sparse) dict.
    
    Parameters
    ----------
    M: dict
        The dictionary defining the entries M_ij of the matrix in the form:
        M[(i, j)] = M_ij
        
    n_rows: int, optional
        The number of rows.

    n_cols: int, optional
        The number of columns.
    
    symmetry: int, optional
        If 0, no symmetry is assumed (default). 
        If 1, matrix is assumed to be symmetric. Requires n_rows == n_cols.
        If -1, matrix is assumed to be anti-symmetric. Requires n_rows == n_cols.
        
    Returns
    -------
    A: ndarray
        A numpy ndarray representing the requested matrix.
    '''
    assert symmetry in [-1, 0, 1]

    dict_shape = max(M.keys(), default=(0, 0))
    n_rows = kwargs.get('n_rows', dict_shape[0] + 1)
    n_cols = kwargs.get('n_cols', dict_shape[1] + 1)
    
    # create a column-matrix
    if symmetry == 0:
        mat = [[0]*n_rows for k in range(n_cols)]
        for i in range(n_rows):
            for j in range(n_cols):
                mat[j][i] = M.get((i, j), 0)
    else:
        dim = max([n_rows, n_cols])
        mat = [[0]*dim for k in range(dim)]
        for i in range(dim):
            for j in range(i + 1):
                hij = M.get((i, j), 0)
                hji = M.get((j, i), 0)
                if hij != 0 and hji != 0:
                    assert hij == symmetry*hji
                if hij == 0 and hji != 0:
                    hij = symmetry*hji
                # (hij != 0 and hji == 0) or (hij == 0 and hji == 0). 
                mat[j][i] = hij
                mat[i][j] = symmetry*hij
    return np.array(mat).transpose()
    
def vecmat(mat):
    '''
    Map a given NxN-matrix to a vector
    '''
    return np.concatenate([mat[k, :] for k in range(mat.shape[0])])

def matvec(vec):
    '''
    Map a given vector of length N**2 to an NxN matrix. This map is the
    inverse of the vecmat routine.
    '''
    n = len(vec)
    assert np.sqrt(n)%1 == 0, 'Vector does not appear to originate from square matrix.'
    m = int(np.sqrt(n))
    return np.array([[vec[j + k*m] for j in range(m)] for k in range(m)])

def adjoint(mat):
    '''
    Map a given NxN-matrix to its adjoint representation with respect to the vecmat and matvec routines.
    '''
    assert mat.shape[0] == mat.shape[1], 'Matrix not square.'
    n = mat.shape[0]
    delta = lambda *z: 1 if z[0] == z[1] else 0
    result = np.zeros([n**2, n**2], dtype=np.complex128)
    for u in range(n**2):
        alpha, beta = divmod(u, n)
        for v in range(n**2):
            i, j = divmod(v, n)
            result[v, u] = mat[i, alpha]*delta(beta, j) - delta(alpha, i)*mat[beta, j]      
    return result

def create_pq2xieta(dim, code, **kwargs):
    '''
    Create a unitary matrix, mapping (p, q)-coordinates to (xi, eta)-coordinates via
    xi = (q + 1j*p)/sqrt(2)
    eta = (q - 1j*p)/sqrt(2)
    '''
    assert dim%2 == 0
    
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        if 'dps' in kwargs.keys():
            mp.mp.dps = kwargs['dps'] 
        sqrt2 = mp.sqrt(2)
        
    U1, U2 = [], []
    dim_half = dim//2
    for k in range(dim_half):
        k2 = k + dim_half
        U1.append([0 if i != k and i != k2 else 1/sqrt2 for i in range(dim)])
        U2.append([0 if i != k and i != k2 else 1j/sqrt2 if i == k else -1j/sqrt2 if i == k2 else 0 for i in range(dim)])
        
    out = np.array(U1 + U2).transpose()
    if code == 'mpmath':
        out = mp.matrix(out)
    return out

# This file collects routines which are focused on representing or generating a matrix and to perform
# fundamental matrix operations in various codes.

import numpy as np
import mpmath as mp

def get_package_name(x):
    '''
    Routine intended to get the package name of a specific object (taken from njet.functions).
    
    Parameters
    ----------
    x: obj
        The object to be examined.
        
    Returns
    -------
    str
        A string denoting the code to be used on the object. 
    '''
    return str(x.__class__.__mro__[0].__module__).split('.')[0]
    
def printmat(M, tol=1e-14):
    # print a matrix (for e.g. debugging reasons)
    M = mp.matrix(M)
    mp.nprint(mp.chop(M, tol))
    
def column_matrix_2_code(M, code, **kwargs):
    # translate a list of column vectors to a numpy or mpmath matrix
    if code == 'numpy':
        return np.array(M).transpose()
    if code == 'mpmath':
        return mp.matrix(M).transpose()

def create_J(dim: int):
    r'''
    Create a 2*dim-square matrix J, represented in form of a list of column vectors,
    corresponding to the standard symplectic block-matrix
    
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
    return J1 + J2 


def expandingSum(dim):
    '''Compute a transformation matrix T by which we can transform a given
    (2n)x(2n) matrix M, represented in (q1, p1, q2, p2, ..., qn, pn)-coordinates, into
    a (q1, q2, ..., qn, p1, p2, ..., pn)-representation via
    M' = T^(-1)*M*T. T will be orthogonal, i.e. T^(-1) = T.transpose().
    
    See also Refs. [1, 2] or (alternatively) in Ref. [3], p. 292., here. In particular, M
    is given in terms of 2x2 block matrices, then M' is called the 'expanding Sum' of M. This explains the name
    of this routine.
    
    Parameters
    ----------
    dim: int
        number of involved coordinates (i.e. 2*dim dimension of phase space)
        
    Returns
    -------
    np.matrix
        Numpy matrix T defining the aforementioned transformation.
        
    Reference(s):
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonances for the CERN PS and SPS", PhD Thesis (2019)
    [2]: M. Titze: "On emittance and optics calculation from the tracking data in periodic lattices", arXiv.org (2019)
    [3]: R. J. de la Cruz and H. Faßbender: "On the diagonalizability of a matrix by a symplectic equivalence, similarity or congruence transformation (2016).
    '''
    dim2 = dim*2
    T = np.zeros([dim2, dim2])
    # define the columns of T:
    for j in range(dim2):
        if j%2 == 0: # note that j starts from 0, in contrast to what is given in Refs. [1, 2]. Therefore the conditions are reversed.
            T[(j + 1)//2, j] = 1
        else:
            T[dim + j//2, j] = 1
    return T.transpose()


def matrix_from_dict(M, code, symmetry: int=0, **kwargs):
    
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
        
    code: str
        Requested code passed to 'column_matrix_2_code' routine.
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
    return column_matrix_2_code(mat, code=code)


class cmat: # TODO: May work on a class to conveniently switch between numpy and mpmath code.
    '''
    Class to model a (sparse) matrix for various codes. 
    '''
    def __init__(self, M, **kwargs):
        # M is assumed to be a dictionary, mapping tuples of indices to values.
        # This means that M can be sparsely defined, but then a 'shape' argument should be provided.
        self.entries = M
        self.rows, self.columns = [], []
        for i, j in self.entries.keys():
            self.rows.append(i)
            self.columns.append(j)
        self.shape = kwargs.get('shape', (max(self.rows) + 1, max(self.columns) + 1))
        
    def tolist(self):
        return [[self.entries.get((i, j), 0) for j in range(self.shape[1])] for i in range(self.shape[0])]
        
    def transpose(self):
        result = {(i, j): self.entries[(j, i)] for j, i in self.entries.keys()}
        return self.__class__(result, shape=(self.shape[1], self.shape[0]))
    
    def conjugate(self):
        result = {tpl: self.entries[tpl].conjugate() for tpl in self.entries.keys()}
        return self.__class__(result, shape=self.shape)
    
    def adjoint(self):
        return self.transpose().conjugate()
    
    def diagonal(self):
        return [self.entries.get((k, k), 0) for k in range(max(self.shape))]
        
    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        result = {}
        common_indices = set(self.columns).intersection(set(other.rows))
        for i in self.rows:
            for k in other.columns:
                result[(i, k)] = sum([self.entries.get((i, j), 0)*other.entries.get((j, k), 0) for j in common_indices])
        return self.__class__(result, shape=(self.shape[0], other.shape[1]))
    
    def __add__(self, other):
        result = {}
        if not isinstance(self, type(other)):
            # add value to every entry
            for k in self.entries.keys():
                sum_value = self.entries[k] + other
                if sum_value != 0:
                    result[k] = sum_value
        else:
            assert self.shape == other.shape
            for k in set(self.entries.keys()).union(set(other.entries.keys())):
                sum_value = self.entries.get(k, 0) + other.entries.get(k, 0)
                if sum_value != 0:
                    result[k] = sum_value
        return self.__class__(result, shape=self.shape)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self.__class__({k: -v for k, v in self.entries.items()}, shape=self.shape)
    
    def __sub__(self, other):
        return self + -other
    
    def __str__(self):
        return repr(self.tolist()) # TMP

    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>'
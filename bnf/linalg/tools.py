# This script collects (or loads) routines which are more fundamental and are
# often required for more sophisticated routines.

import numpy as np
import mpmath as mp
from scipy.linalg import schur

from .checks import relative_eq 

def twonorm(vector, code='numpy', mode='complex', **kwargs):
    # Compute the 2-norm of a vector.
    # This seems to provide slightly faster results than np.linalg.norm
    if mode == 'complex':
        sum2 = sum([vector[k].conjugate()*vector[k] for k in range(len(vector))])
    else:
        sum2 = sum([vector[k]*vector[k] for k in range(len(vector))])
    if code == 'numpy':
        return np.sqrt(sum2)
    if code == 'mpmath':
        return mp.sqrt(sum2)    
    
def gram_schmidt(vectors, mode='complex', **kwargs):
    '''Gram-Schmidt orthogonalization procedure of linarly independent vectors with complex entries, i.e.
    'unitarization'.
    
    Parameters
    ----------
    vectors: list
        list of vectors to be orthogonalized.
        
    mode: str, optional
        If mode == 'complex' (default), then all norms and scalar products are computed using conjugation.
        
    **kwargs 
        Additional arguments passed to linalg.twonorm
    
    Returns
    -------
    list
        list of length len(vectors) which are mutually unitary. I.e.
        with O := np.array(list).T it holds O^H@O = 1, where ^H means transposition and complex conjugation.
    '''
    k = len(vectors)
    dim = len(vectors[0])
    norm_0 = twonorm(vectors[0], mode=mode, **kwargs)
    ortho = {(m, 0): vectors[0][m]/norm_0 for m in range(dim)} # ortho[(m, j)] denotes the m-th component of the j-th (new) vector
    for i in range(1, k):
        sum1 = {(m, i): vectors[i][m] for m in range(dim)}
        for j in range(i):
            if mode == 'complex':
                scalar_product_ij = sum([ortho[(r, j)].conjugate()*vectors[i][r] for r in range(dim)])
            else:
                scalar_product_ij = sum([ortho[(r, j)]*vectors[i][r] for r in range(dim)])
                
            for m in range(dim):
                sum1[(m, i)] -= scalar_product_ij*ortho[(m, j)]  
        norm_i = twonorm([sum1[(m, i)] for m in range(dim)], mode=mode, **kwargs)
        for m in range(dim):
            ortho[(m, i)] = sum1[(m, i)]/norm_i
    return [[ortho[(m, i)] for m in range(dim)] for i in range(k)]


def rref(M, augment=None):
    '''
    Compute the reduced row echelon form of M (M can be a real or complex matrix).
    '''
    # reduced row echelon form of M
    n, m = M.shape
    if augment == None:
        augment = np.eye(n)
    assert augment.shape[1] == n
    Mone = np.bmat([M, augment])
    
    # transform Mone = (M | 1)
    pivot_row_index = 0
    next_pivot_row_index = 1
    pivot_indices = [] # to record the pivot indices
    for j in range(m):
        column_j_has_pivot = False
        for k in range(pivot_row_index, n):
            # skip 0-entries
            if Mone[k, j] == 0:
                continue
            Mone[k, :] = Mone[k, :]/Mone[k, j]        
            # exchange the first non-zero entry with those at the top (if it is not already the top)
            if not column_j_has_pivot:
                if k > pivot_row_index:
                    pivot_row = np.copy(Mone[pivot_row_index, :])
                    Mone[pivot_row_index, :] = np.copy(Mone[k, :])
                    Mone[k, :] = pivot_row
                pivot_indices.append((pivot_row_index, j)) # also record the pivot indices for output
                next_pivot_row_index = pivot_row_index + 1
                column_j_has_pivot = True
                continue
            # eliminate k-th row
            Mone[k, :] = Mone[k, :] - Mone[pivot_row_index, :]
            
        # for the reduced form, we also need to remove entries from the rows < pivot:
        if not column_j_has_pivot:
            continue
        for k in range(pivot_row_index):
            # skip 0-entries
            if Mone[k, j] == 0:
                continue
            Mone[k, :] = Mone[k, :] - Mone[k, j]*Mone[pivot_row_index, :]
        
        pivot_row_index = next_pivot_row_index
        
    return Mone[:,:m], Mone[:,m:], pivot_indices


def imker(M):
    '''
    Obtain a basis for the image and the kernel of M.
    M can be a real or complex matrix.
    '''
    # Idea taken from
    # https://math.stackexchange.com/questions/1612616/how-to-find-null-space-basis-directly-by-matrix-calculation
    ImT, KerT, pivots = rref(M.transpose())
    zero_row_indices = pivots[-1][0] + 1
    kernel = KerT[zero_row_indices:, :].transpose()
    image = ImT[:zero_row_indices, :].transpose()
    return image, kernel


def basis_extension(*vects, gs=False, **kwargs):
    '''
    Provide an extension of a given set of vectors
    to span the full space. The vectors can have real or
    complex coefficients.
    '''
    _, ext = imker(np.array(vects).conjugate())
    n, m = ext.shape
    if gs and n > 0 and m > 0:
        ext = np.array(gram_schmidt([[ext[k, j] for k in range(n)] for j in range(m)], **kwargs)).transpose()
    return ext


def eigenspaces(M, code='numpy', flatten=False, **kwargs):
    '''
    Let M be a square matrix. Then this routine will determine a basis of normalized eigenvectors. Hereby
    eigenvectors belonging to the same eigenvalues are (complex) orthogonalized.
    
    Parameters
    ----------
    M:
        list of vectors defining a (n x n)-matrix.
        
    code: str, optional
        Code to be used to determine the eigenvalues and eigenvectors. 
        Currently supported: 'numpy' (default), 'mpmath'.
        
    flatten: boolean, optional
        If True, flatten the respective results (default: False).
        
    Returns
    -------
    eigenvalues: list
        List of elements, where the k-th element constitute the eigenvalue to the k-th eigenspace.
    
    eigenvectors: list
        List of lists, where the k-th element is a list of pairwise unitary vectors spanning the k-th eigenspace.
    '''
    if code == 'numpy':
        eigenvalues, eigenvectors = np.linalg.eig(M)
        eigenvalues = eigenvalues.tolist()
        eigenvectors = eigenvectors.T.tolist()
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32) # number of digits defining precision; as default we set 32.
        eigenvalues, eigenvectors = mp.eig(mp.matrix(M))
        eigenvectors = [[eigenvectors[k, j] for k in range(len(eigenvalues))] for j in range(len(eigenvalues))]
        
    n = len(eigenvalues)
    assert n > 0
    
    # group the indices of the eigenvectors if they belong to the same eigenvalues.
    eigenspaces = [[0]] # 'eigenspaces' will be the collection of these groups of indices.
    for i in range(1, n):
        j = 0
        while j < len(eigenspaces):
            if relative_eq(eigenvalues[i], eigenvalues[eigenspaces[j][0]], **kwargs):
                eigenspaces[j].append(i)
                break
            j += 1    
        if j == len(eigenspaces):
            eigenspaces.append([i])
                
    # orthogonalize vectors within the individual eigenspaces
    eigenvalues_result, eigenvectors_result = [], []
    for indices in eigenspaces:
        vectors = [eigenvectors[k] for k in indices]
        on_vectors = gram_schmidt(vectors, code=code, **kwargs)
        on_eigenvalues = [eigenvalues[k] for k in indices]
        if flatten:
            eigenvectors_result += on_vectors
            eigenvalues_result += on_eigenvalues
        else:
            eigenvectors_result.append(on_vectors)
            eigenvalues_result.append(on_eigenvalues[0]) # all of these eigenvalues are considered to be equal, so we pic the first one.
    return eigenvalues_result, eigenvectors_result


def get_principal_sqrt(M, **kwargs):
    '''
    A numeric algorithm to obtain the principal square root of a matrix M (if it exists).
    The principal square is the unique square root of a matrix so that -pi/2 < arg(l(X)) <= pi/2
    holds, where
    It can can be expressed in terms of a polynomial in M.
    Refernce: Ake Björk: Numerical Methods in Matrix Computations (2014), sec. 3.8.1.
    '''
    T, Q = schur(M, **kwargs) # Q@T@Q.transpose().conjugate() = M 
    # N.B. Q is a unitary matrix
    
    dim = len(T)
    S = np.diag(np.sqrt(T.diagonal(), dtype=complex))
    for diagonal in range(1, dim): # diagonal = 1: first off-diagonal S[0, 1], S[1, 2], ... etc.
        for diagonal_index in range(dim - diagonal):
            i = diagonal_index
            j = diagonal + diagonal_index        
            S[i, j] = (T[i, j] - sum([S[i, k]*S[k, j] for k in range(i + 1, j)]))/(S[i, i] + S[j, j]) # the j - 1 in the Eq. in the reference must be increased by 1 here due to Python convention.

    # check: S@S = T
    return Q@S@Q.transpose().conjugate()

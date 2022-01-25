# This script collects (or loads) routines which are more fundamental and are
# often required for more sophisticated routines.

import numpy as np
import mpmath as mp
from scipy.linalg import schur

from .matrix import get_package_name

def twonorm(vector, mode='complex', code='numpy', **kwargs):
    # Compute the 2-norm of a vector.
    # This seems to provide slightly faster results than np.linalg.norm
    if mode == 'complex':
        sum2 = sum([vector[k].conjugate()*vector[k] for k in range(len(vector))])
    else:
        sum2 = sum([vector[k]*vector[k] for k in range(len(vector))])

    if code == 'numpy':
        return np.sqrt(np.real(sum2))
    if code == 'mpmath':
        return mp.sqrt(mp.re(sum2))    
    
def gram_schmidt(vectors, mode='complex', tol=1e-15, **kwargs):
    '''Gram-Schmidt orthogonalization procedure of linarly independent vectors with complex entries, i.e.
    'unitarization'.
    
    Parameters
    ----------
    vectors: list
        list of vectors to be orthogonalized.
        
    mode: str, optional
        If mode == 'complex' (default), then all norms and scalar products are computed using conjugation.
        
    **kwargs 
        Additional arguments passed to .twonorm
    
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
        if abs(norm_i) < tol:
            raise RuntimeError(f'Division by zero with mode ({mode}) encountered; check input on linearly independence.')
        for m in range(dim):
            ortho[(m, i)] = sum1[(m, i)]/norm_i
    return [[ortho[(m, i)] for m in range(dim)] for i in range(k)]


def rref(M, augment=None, tol=1e-10, **kwargs):
    '''
    Compute the reduced row echelon form of M (M can be a real or complex matrix).
    
    Parameters
    ----------
    M: matrix
        matrix to be transformed.
        
    augment: matrix, optional
        optional matrix to be used on the right-hand side of M, and which will be simultaneously transformed.
        If nothing specified, the identity matrix will be used.
        
    tol: float, optional
        A tolerance by which we identify small numbers as zero.
    
    Returns
    -------
    triple
        A triple consisting of i) the transformed matrix M, ii) the transformed augment and iii) the pivot indices
        of the transformed matrix M (a list of tuples).
    '''
    code = get_package_name(M)
    # reduced row echelon form of M
    if code == 'numpy':
        n, m = M.shape
        if augment == None:
            augment = np.eye(n)        
        assert augment.shape[0] == n
        Mone = np.bmat([M, augment])
        
    elif code == 'mpmath':
        n, m = M.rows, M.cols
        if augment == None:
            augment = mp.eye(n)
        assert augment.rows == n
        Mone = mp.zeros(n, m + augment.cols)
        Mone[:, :m] = M
        Mone[:, m:] = augment
    
    # transform Mone = (M | 1)
    pivot_row_index = 0
    pivot_indices = [] # to record the pivot indices
    for j in range(m):
        
        # Step 1: determine the pivot row of column j
        column_j_has_pivot = False
        for k in range(pivot_row_index, n):
            # skip 0-entries
            if abs(Mone[k, j]) < tol:
                continue
                
            if k > pivot_row_index:
                # exchange this entry with those at the intended pivot_row_index row
                if code == 'numpy':
                    pivot_row = np.copy(Mone[pivot_row_index, :])
                    Mone[pivot_row_index, :] = np.copy(Mone[k, :])/Mone[k, j]
                elif code == 'mpmath':
                    pivot_row = Mone[pivot_row_index, :].copy()
                    Mone[pivot_row_index, :] = Mone[k, :].copy()/Mone[k, j]
                Mone[k, :] = pivot_row

            # normalize the column belonging to the pivot index
            Mone[pivot_row_index, :] = Mone[pivot_row_index, :]/Mone[pivot_row_index, j]
                    
            pivot_indices.append((pivot_row_index, j)) # record the pivot indices for output
            column_j_has_pivot = True
            break
                
        if not column_j_has_pivot: 
            # It can happen that column_j_has_pivot = False: Namely, if the colum has only zero entries.
            # In this case we just proceed with the next column.
            continue
            
        # Step 2: eliminate the other non-zero rows.
        for k in range(0, n):
            if abs(Mone[k, j]) < tol or k == pivot_row_index:
                continue
            Mone[k, :] = Mone[k, :] - Mone[k, j]*Mone[pivot_row_index, :]
            
        pivot_row_index = pivot_row_index + 1
        
    return Mone[:,:m], Mone[:,m:], pivot_indices


def imker(M, **kwargs):
    '''
    Obtain a basis for the image and the kernel of M.
    M can be a real or complex matrix.
    
    Parameters
    ----------
    M:
        matrix to be analyzed.
        
    **kwargs
        Additional arguments passed to rref routine.
        
    Returns
    -------
    image:
        matrix spanning the image of M
        
    kernel:
        matrix spanning to the kernel of M
    '''
    # Idea taken from
    # https://math.stackexchange.com/questions/1612616/how-to-find-null-space-basis-directly-by-matrix-calculation    

    code = kwargs.get('code', get_package_name(M))
    if code == 'numpy':
        M = np.array(M)
    if code == 'mpmath':
        M = mp.matrix(M)
    
    ImT, KerT, pivots = rref(M.transpose(), **kwargs) # transpose the input matrix to obtain kernel & image in the end.
    if len(pivots) == 0:
        # this can happen if M = 0, so no pivot points exist.
        if code == 'numpy':
            kernel = np.eye(M.shape[1])
        elif code == 'mpmath':
            kernel = mp.eye(M.cols)
        image = 0*M
    else: 
        zero_row_indices = pivots[-1][0] + 1
        kernel = KerT[zero_row_indices:, :].transpose()
        image = ImT[:zero_row_indices, :].transpose()
    return image, kernel


def basis_extension(*vects, gs=False, **kwargs):
    '''
    Provide an extension of a given set of vectors to span the full space. 
    The vectors can have real or complex coefficients.
    
    Parameters
    ----------
    *vects:
        Vectors to be extended
        
    gs: boolean, optional
        Apply the Gram-Schmidt orthogonalization procedure on the extension.
        
    **kwargs
        Optional arguments passed to gram_schmidt. In use only if gs=True.
        
    Returns
    -------
    matrix
        A matrix representing a basis extension of the given vectors.
    '''
    code = get_package_name(vects[0])
    if code == 'numpy':
        vects = np.array(vects)
    if code == 'mpmath':
        vects = mp.matrix(vects)
        
    _, ext = imker(vects.conjugate(), **kwargs)
    
    if code == 'numpy':
        n, m = ext.shape
    elif code == 'mpmath':
        n, m = ext.rows, ext.cols
        
    if gs and n > 0 and m > 0:
        ext = gram_schmidt([[ext[k, j] for k in range(n)] for j in range(m)], **kwargs)
        
    if code == 'numpy':
        return np.array(ext).transpose()
    elif code == 'mpmath':
        return mp.matrix(ext).transpose()


def eig(M, **kwargs):
    '''
    Compute the eigenvalues and eigenvectors of a given matrix, based on underlying code.
    
    Parameters
    ----------
    M: matrix
        Matrix to be considered.
    '''
    code = get_package_name(M)
    if code == 'numpy':
        eigenvalues, eigenvectors = np.linalg.eig(M)
        eigenvalues = eigenvalues.tolist()
        eigenvectors = eigenvectors.T.tolist()
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32) # number of digits defining precision.
        eigenvalues, eigenvectors = mp.eig(mp.matrix(M))
        eigenvectors = [[eigenvectors[k, j] for k in range(len(eigenvalues))] for j in range(len(eigenvalues))]
    return eigenvalues, eigenvectors


def eigenspaces(M, flatten=False, tol=1e-10, check=True, **kwargs):
    '''
    Let M be a square matrix. Then this routine will determine a basis of normalized eigenvectors. Hereby
    eigenvectors belonging to the same eigenvalues are (complex) orthogonalized.
    
    Parameters
    ----------
    M:
        list of vectors defining a (n x n)-matrix.
        
    flatten: boolean, optional
        If True, flatten the respective results (default: False).
        
    tol: float, optional
        Parameter to identify small values as being zero.
        
    check: boolean, optional
        Check if the number of zero-eigenvalues is consistent with the dimension of the kernel of the input matrix within the given tolerance.
        The kernel of the input matrix is hereby determined by the imker routine.
        
    Returns
    -------
    eigenvalues: list
        List of elements, where the k-th element constitute the eigenvalue to the k-th eigenspace.
    
    eigenvectors: list
        List of lists, where the k-th element is a list of pairwise unitary vectors spanning the k-th eigenspace.
    '''
    code = get_package_name(M)
    eigenvalues, eigenvectors = eig(M, **kwargs)
        
    n = len(eigenvalues)
    assert n > 0
    dim = len(eigenvectors[0])
        
    # group the indices of the eigenvectors if they belong to the same eigenvalues.
    eigenspaces = [[0]] # 'eigenspaces' will be the collection of these groups of indices.
    for i in range(1, n):
        j = 0
        while j < len(eigenspaces):
            if abs(eigenvalues[i] - eigenvalues[eigenspaces[j][0]]) < tol: 
                # eigenvalues[i] has been identified to belong to eigenspaces[j] group; append the index to this group
                eigenspaces[j].append(i)
                break
            j += 1    
        if j == len(eigenspaces): 
            # no previous eigenspace belonging to this eigenvalue has been found; create a new group
            eigenspaces.append([i])
    
    if check:
        # check if we have identified the number of zero-eigenvalues 
        # to agree with the dimension of the kernel of the input matrix
        image, kernel = imker(M, tol=tol, **kwargs)
        if code == 'numpy':
            dim_kernel = kernel.shape[1]
        if code == 'mpmath':
            dim_kernel = kernel.cols
        # check if tolerance can detect the zero-eigenvalues
        n_zero_eigenvalues = len([e for e in eigenspaces if abs(eigenvalues[e[0]]) < tol])
        assert dim_kernel == n_zero_eigenvalues, f'The number {n_zero_eigenvalues} of zero-eigenvalues is not consistent with the dimension {dim_kernel} of the kernel of the input matrix, both determined using a tolerance of: {tol}.'
                
    # orthogonalize vectors within the individual eigenspaces
    eigenvalues_result, eigenvectors_result = [], []
    for indices in eigenspaces:
        vectors = [[eigenvectors[k][j] for k in indices] for j in range(dim)]
        # the vectors given by the eig routine may be linearly dependent; we therefore orthogonalize its image
        vimage, vkernel = imker(vectors, tol=tol, code=code)
        vimage = vimage.transpose().tolist() # transpose().tolist() creates a list of column-vectors, as required by gram_schmidt routine
        
        basis_e = [v for v in vimage if twonorm(v, **kwargs) >= tol]
        zeros_e = [v for v in vimage if twonorm(v, **kwargs) < tol]
        if len(basis_e) > 0:
            basis_e = gram_schmidt(basis_e, tol=tol, **kwargs)
        on_vectors = zeros_e + basis_e
        on_eigenvalues = [eigenvalues[k] for k in indices[:len(on_vectors)]]
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

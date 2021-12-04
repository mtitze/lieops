import numpy as np
import mpmath as mp


def printdb(M, tol=1e-14):
    # print a matrix (for debugging reasons)
    M = mp.matrix(M)
    mp.nprint(mp.chop(M, tol))
    

def twonorm(vector, code='numpy'):
    # Compute the 2-norm of a vector.
    # This seems to provide slightly faster results than np.linalg.norm
    sum2 = sum([vector[k].conjugate()*vector[k] for k in range(len(vector))])
    if code == 'numpy':
        return np.sqrt(sum2)
    if code == 'mpmath':
        return mp.sqrt(sum2)
    
    
def almosteq(a, b, tol=1e-14, **kwargs):
    '''
    Check if the relative difference between two values is smaller than a given value tol.
    
    Parameters
    ----------
    a: complex
        First parameter
    
    b: complex
        Second parameter
        
    tol: float, optional
        Tolerance below which we consider the relative difference of the two parameters as 
        zero.
        
    Returns
    -------
    boolean
    '''
    if a == 0 and b == 0:
        return True
    else:
        return abs(a - b)/max([abs(a), abs(b)]) < tol
    
    
def column_matrix_2_code(M, code='numpy', **kwargs):
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
        
    **kwargs
        Arguments passed to 'column_matrix_2_code' routine.
    '''
    assert symmetry in [-1, 0, 1]

    dict_shape = max(M.keys())
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
    return column_matrix_2_code(mat, **kwargs)
    

def complex_gram_schmidt(vectors, **kwargs):
    '''Gram-Schmidt orthogonalization procedure of linarly independent vectors with complex entries, i.e.
    'unitarization'.
    
    Parameters
    ----------
    vectors: list
        list of vectors to be orthogonalized.
        
    **kwargs are passed to linalg.twonorm
    
    Returns
    -------
    list
        list of length len(vectors) which are mutually unitary. I.e.
        with O := np.array(list).T it holds O^H@O = 1, where ^H means transposition and complex conjugation.
    '''
    k = len(vectors)
    dim = len(vectors[0])
    norm_0 = twonorm(vectors[0], **kwargs)
    ortho = {(m, 0): vectors[0][m]/norm_0 for m in range(dim)} # ortho[(m, j)] denotes the m-th component of the j-th (new) vector
    for i in range(1, k):
        sum1 = {(m, i): vectors[i][m] for m in range(dim)}
        for j in range(i):
            scalar_product_ij = sum([ortho[(r, j)].conjugate()*vectors[i][r] for r in range(dim)])
            for m in range(dim):
                sum1[(m, i)] -= scalar_product_ij*ortho[(m, j)]  
        norm_i = twonorm([sum1[(m, i)] for m in range(dim)], **kwargs)
        for m in range(dim):
            ortho[(m, i)] = sum1[(m, i)]/norm_i
    return [[ortho[(m, i)] for m in range(dim)] for i in range(k)]


def rref(M, augment=None):
    '''
    Compute the reduced row echelon form of M. M can be a real or complex matrix.
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
    ImT, KerT, pivots = rref(M.transpose())
    zero_row_indices = pivots[-1][0] + 1
    kernel = KerT[zero_row_indices:, :].transpose()
    image = ImT[:zero_row_indices, :].transpose()
    return image, kernel


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
            if almosteq(eigenvalues[i], eigenvalues[eigenspaces[j][0]], **kwargs):
                eigenspaces[j].append(i)
                break
            j += 1    
        if j == len(eigenspaces):
            eigenspaces.append([i])
                
    # orthogonalize vectors within the individual eigenspaces
    eigenvalues_result, eigenvectors_result = [], []
    for indices in eigenspaces:
        vectors = [eigenvectors[k] for k in indices]
        on_vectors = complex_gram_schmidt(vectors, code=code)
        on_eigenvalues = [eigenvalues[k] for k in indices]
        if flatten:
            eigenvectors_result += on_vectors
            eigenvalues_result += on_eigenvalues
        else:
            eigenvectors_result.append(on_vectors)
            eigenvalues_result.append(on_eigenvalues[0]) # all of these eigenvalues are considered to be equal, so we pic the first one.
    return eigenvalues_result, eigenvectors_result


def anti_diagonalize_skew(M, code='numpy', **kwargs):
    r'''Anti-diagonalize a real skew symmetric matrix A so that it will have the block-form
    
             /  0   X  \
        A =  |         |
             \ -X   0  /
    
    where X is a diagonal matrix with positive entries.
    
    Attention: No check is made if the given matrix has real coefficients or is skew symmetric.
    
    Parameters
    ----------
    M:
        list of vectors defining a real skew symmetric (n x n)-matrix.

    **kwargs 
        Additional arguments are passed to 'almosteq' routine. Warning: If chosing 'mpmath' as code and dps value is
        set too small (corresponding to a too rough precision), then the tolerance 'tol' may have to be increased as well.
    
    Returns
    -------
    matrix
        Orthogonal real matrix R so that R.transpose()@M@R has skew-symmetric antidiagonal form with respect 
        to (n x n) block matrices (Hereby M denotes the input matrix).
    '''
    evalues, evectors = eigenspaces(M, flatten=True, code=code, **kwargs)
    n = len(evalues)
    
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32)
        sqrt2 = mp.sqrt(2)
    
    # now construct a real basis
    v_block1, v_block2 = [], []
    processed_indices = []
    for i in range(1, n):
        if i in processed_indices:
            continue
        for j in range(i):
            if j in processed_indices:
                continue
            # pic those pairs of eigenvalues which belong to the same 'plane':
            same_plane = almosteq(evalues[i].imag, -evalues[j].imag, **kwargs)
            if not same_plane:
                continue
                
            processed_indices.append(i)
            processed_indices.append(j)
            
            # Select the index belonging to the eigenvalue with positive imaginary part
            # Attention: This sets the signature of the matrix Omega in Eq. (6) in Williamson decomposition.
            # We want that the top right entry is positive and the bottom left entry is negative. Therefore:
            if evalues[i].imag >= 0: # the imaginary part of the eigenvalue of aj + 1j*bi (see below) will be positive
                pos_index = i
            else: # change the role of i and j (orientation) to maintain that ai + 1j*bi (see below) is positive; evalues[j].imag > 0
                pos_index = j
            
            ai = [(evectors[pos_index][k] + evectors[pos_index][k].conjugate())/sqrt2 for k in range(n)] # the 1/sqrt2-factor comes from the fact that evectors[pos_index] has been normalized to 1.
            bi = [-1j*(evectors[pos_index][k] - evectors[pos_index][k].conjugate())/sqrt2 for k in range(n)]
            v_block1.append(ai)
            v_block2.append(bi)
    return column_matrix_2_code(v_block1 + v_block2, code=code)


def williamson(V, code='numpy', **kwargs):
    r'''Compute Williamson's decomposition of a symmetric positive definite real matrix,
    according to 'R. Simon, S. Chaturvedi, and V. Srinivasan: Congruences and Canonical Forms 
    for a Positive Matrix: Application to the Schweinler-Wigner Extremum Principle'.
    
    The output matrix S and the diagonal D are satisfying the relation:
    S.transpose()@D@S = V
    
    Attention: No extensive checks if V is actually symmetric positive definite real.
    
    Parameters
    ----------
    V: list
        List of vectors (subscriptables) defining the matrix.
        
    **kwargs
        Additional arguments are passed to 'anti_diagonalize_skew' routine.
    
    Returns
    -------
    S: matrix
        Symplectic matrix with respect to the standard block-diagonal form
        
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
        which diagonalizes V as described above.
        
        Note that results for a different J' can be obtained by applying a matrix congruence operation T.transpose()*S*T to the
        result S, where S is obtained by an input matrix V' by the respective inverse congruence operation on V.
        Here T denotes the transformation satisfying J' = T.transpose()@J@T. 
        
    D: matrix
        The diagonal matrix as described above.    
    '''
    if code == 'numpy':
        evalues, evectors = np.linalg.eigh(V)
        sqrtev = np.sqrt(evalues)
        diag = np.diag(sqrtev)
        diagi = np.diag(1/sqrtev)
        evectors = np.array(evectors)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32)
        V = mp.matrix(V)
        evalues, evectors = mp.eigh(V)
        diag = mp.diag([mp.sqrt(e) for e in evalues])
        diagi = mp.diag([1/mp.sqrt(e) for e in evalues])

    assert all([e > 0 for e in evalues]), f'Input matrix eigenvalues of matrix\n{V}\nnot all positive.'
    
    V12 = evectors@diag@evectors.transpose() # V12 means V^(1/2), the square root of V.
    V12i = evectors@diagi@evectors.transpose()
    
    dim2 = len(V12)
    assert dim2%2 == 0
    dim = dim2//2
    
    J = column_matrix_2_code(create_J(dim), code=code)    
    skewmat = V12i@J@V12i
    A = anti_diagonalize_skew(skewmat, code=code, **kwargs)    
    K = A.transpose()@skewmat@A # the sought anti-diagonal matrix

    # obtain D as described in the reference above
    Di_values = [K[i, i + dim] for i in range(dim)]*2
    D = [1/e for e in Di_values]
    if code == 'numpy':
        D12i = np.array(np.diag([np.sqrt(e) for e in Di_values]))
        D = np.array(np.diag(D))
    if code == 'mpmath':
        D12i = mp.matrix(mp.diag([mp.sqrt(e) for e in Di_values]))
        D = mp.matrix(mp.diag(D))
    S = D12i@A.transpose()@V12
    return S, D

    
def normal_form(H2, T=[], code='numpy', **kwargs):
    r'''
    Perform linear calculations to transform a given second-order Hamiltonian,
    expressed in canonical coordinates (q, p), to
    complex normal form coordinates xi, eta. Along the way, the symplectic linear map to
    real normal form is computed. The quantities xi and eta are defined as in my thesis,
    Chapter 1.
    
    Paramters
    ---------
    H2: matrix
        Real symmetric positive definite Matrix.
        
    T: matrix, optional
        Orthogonal matrix to change the ordering of canoncial
        coordinates and momenta, given here by default as (q1, ..., qn, p1, ..., pn), 
        into a different order. I.e. T transforms the 
        (n x n) block matrix
        
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
        into a matrix J' by matrix congruence: J' = T.transpose()@J@T.
        
    code: str, optional
        The code in which the matrix calculations should be performed. Supported codes: 'numpy', 'mpmath'.
        If the input is given in form of a matrix, the user has to ensure the correct code is selected.
        Default: 'numpy'.
             
    Returns
    -------
    dict
        Dictionary containing various linear maps. The entries of this dictionary are described in the following.
        
        S: The symplectic map diagonalizing H2 via S.transpose()@D@S = H2, where D is a diagonal matrix.
        Sinv: The inverse of S, i.e. the symplectic map to (real) normal form.
        H2: The input in matrix form.
        T: The (optional) matrix T described above.
        J: The (original) symplectic structure J' = J.transpose()@J@T, where J is the block-matrix given above.
        J2: The new symplectic structure for the (xi, eta)-coordinates.
        U: The unitary map from the S(p, q) = (u, v)-block coordinates to the (xi, eta)-coordinates.
        Uinv: The inverse of U.
        K: The linear map transforming (q, p) to (xi, eta)-coordinates. Hence, it will transform
           H2 to complex normal form via Kinv.transpose()*H2*Kinv. K is given by U*S*T.
        Kinv: The inverse of K.
        rnf: The 'real' normal form, by which we understand the diagonalization of H2 relative to its 
            real normalizing coordinates (the matrix D described above).
        cnf: The 'complex' normal form, which is given as the representation of H2 in terms of the complex
            normalizing (xi, eta)-coordinates.
        D: The real entries of rnf in form of a list.
    '''
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32) 
        sqrt2 = mp.sqrt(2)
        
    dim = len(H2)
    assert dim%2 == 0, 'Dimension must be even.'
        
    # Perform symplectic diagonalization
    if len(T) != 0: # transform H2 to default block ordering before entering williamson routine; the results will later be transformed back. This is easier instead of keeping track of orders inside the subroutines.
        H2 = T@H2@T.transpose() 
    S, D = williamson(V=H2, code=code, **kwargs)
    
    # The first dim columns of S denote (new) canonical coordinates u, the last dim columns of S
    # denote (new) canonical momenta v. The block-matrix U transforming the block-vector (u, v) to
    # (xi, eta) (as e.g. defined in my thesis) and has the form:
    U1, U2 = [], []
    dim_half = dim//2
    for k in range(dim_half):
        k2 = k + dim_half
        U1.append([0 if i != k and i != k2 else 1/sqrt2 for i in range(dim)])
        U2.append([0 if i != k and i != k2 else 1j/sqrt2 if i == k else -1j/sqrt2 if i == k2 else 0 for i in range(dim)])
    U = column_matrix_2_code(U1 + U2, code=code)
    # U is hermitian, therefore
    Uinv = U.transpose().conjugate()

    J = column_matrix_2_code(create_J(dim_half), code=code)
    # N.B. (p, J*q) = (Sp, J*S*q) = (u, J*v) = (Uinv*U*u, J*Uinv*U*v) = (Uinv*xi, J*Uinv*eta). Thus:
    J2 = Uinv.transpose()@J@Uinv # the new symplectic structure with respect to the (xi, eta)-coordinates (holds also in the case len(T) != 0)
    Sinv = - J@S.transpose()@J
    K = U@S # K(p, q) = (xi, eta)
    Kinv = Sinv@Uinv  # this map will transform to the new (xi, eta)-coordinates via Kinv.transpose()*H2*Kinv

    if len(T) != 0: # transform results back to the requested (q, p)-ordering
        S = T.transpose()@S@T
        Sinv = T.transpose()@Sinv@T
        D = T.transpose()@D@T
        J = T.transpose()@J@T
        H2 = T.transpose()@H2@T
        
        K = K@T
        Kinv = T.transpose()@Kinv
    
    # assemble output
    out = {}
    out['S'] = S 
    out['Sinv'] = Sinv # this symplectic map will diagonalize H2 in its original 
    # (q, p)-coordinates via Sinv.transpose()*H2*Sinv. Sinv (and S) are symplectic wrt. J
    out['H2'] = H2 # the input matrix
    out['rnf'] = D # the diagonal matrix obtained as a result of the symplectic diagonalization of H2
    out['D'] = [D[i, i].real for i in range(len(D))]
    out['T'] = T
    out['J'] = J # the original symplectic structure
    out['J2'] = J2 # the new symplectic structure
    out['U'] = U # the unitary map from the S(p, q)=(u, v)-block coordinates to the (xi, eta)-coordinates
    out['Uinv'] = Uinv
    out['K'] = K # K(q, p) = (xi, eta)
    out['Kinv'] = Kinv
    out['cnf'] = Kinv.transpose()@H2@Kinv # the representation of H2 in (xi, eta)-coordinates
    
    return out
    

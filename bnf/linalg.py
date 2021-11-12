import numpy as np
import mpmath as mp

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
    
    
def column_matrix_2_code(M, code):
    # translate a list of column vectors to a numpy or mpmath matrix
    if code == 'numpy':
        return np.matrix(M).transpose()
    if code == 'mpmath':
        return mp.matrix(M).transpose()
    

def create_J(dim: int):
    '''
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
    

def complex_gram_schmidt(vectors, **kwargs):
    '''Gram-Schmidt orthogonalization procedure of linarly independent vectors with complex entries, i.e.
    'unitarization'.
    
    Parameters
    ----------
    vectors: list
        list of vectors to be orthogonalized.
        
    **kwargs are passed to townorm
    
    Returns
    -------
    list
        list of length len(vectors) which are mutually unitary. I.e.
        with O := np.matrix(list).T it holds O^H*O = 1, where ^H means transposition and complex conjugation.
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


def eigenspaces(M, code='numpy', flatten=False, **kwargs):
    '''
    Let M be a complex diagonalizable matrix. Then this routine will determine a basis of pairwise unitary
    (or orthogonal) eigenvectors.
    
    Parameters
    ----------
    M:
        list of vectors defining a (n x n)-matrix.
        
    code: str, optional
        Code to be used to determine the eigenvalues and eigenvectors. 
        Currently supported: 'numpy', 'mpmath'.
        Default: 'numpy'
        
    flatten: boolean, optional
        If True, flatten the respective results (Default: False)
        
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
                
    # orthogonalize
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
    '''Anti-diagonalize a real skew symmetric matrix A so that it will have the block-form
    
        /  0  X \
    A = |       |
        \ -X  0 /
    
    where X is a diagonal matrix with positive entries.
    
    Attention: No check is made if the given matrix has real coefficients.
    
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
        Matrix correspond to an orthogonal real matrix R so that R^(-1)*M*R has 
        skew-symmetric antidiagonal form, either with respect to (2 x 2) block matrices
        or with respect to (n x n) block matrices (Hereby M denotes the input matrix).
    '''
    evalues, evectors = eigenspaces(M , flatten=True, code=code, **kwargs)
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
            # Attention: This sets the signature of the matrix Omega in Eq. (6) in williamson decomposition.
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
    '''Compute Williamson's decomposition of a symmetric positive definite real matrix,
    according to 'R. Simon, S. Chaturvedi, and V. Srinivasan: Congruences and Canonical Forms 
    for a Positive Matrix: Application to the Schweinler-Wigner Extremum Principle'.
    
    The output matrix S and the diagonal D are satisfying the relation:
    S.transpose()*D*S = V
    
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
        Here T denotes the transformation satisfying J' = T.transpose()*J*T. 
        
    D: matrix
        The diagonal matrix as described above.    
    '''
    if code == 'numpy':
        evalues, evectors = np.linalg.eigh(V)
        sqrtev = np.sqrt(evalues)
        diag = np.diag(sqrtev)
        diagi = np.diag(1/sqrtev)
        evectors = np.matrix(evectors)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32)
        V = mp.matrix(V)
        evalues, evectors = mp.eigh(V)
        diag = mp.diag([mp.sqrt(e) for e in evalues])
        diagi = mp.diag([1/mp.sqrt(e) for e in evalues])

    assert all([e > 0 for e in evalues]), f'Input matrix eigenvalues of matrix\n{V}\nnot all positive.'
    
    V12 = evectors*diag*evectors.transpose() # V12 means V^(1/2), the square root of V.
    V12i = evectors*diagi*evectors.transpose()
    
    dim2 = len(V12)
    assert dim2%2 == 0
    dim = dim2//2
    
    J = column_matrix_2_code(create_J(dim), code=code)    
    skewmat = V12i*J*V12i
    A = anti_diagonalize_skew(skewmat, code=code, **kwargs)    
    K = A.transpose()*skewmat*A # the sought anti-diagonal matrix
    
    # obtain D as described in the reference above
    Di_values = [K[i, i + dim] for i in range(dim)]*2
    D = [1/e for e in Di_values]
    if code == 'numpy':
        D12i = np.matrix(np.diag([np.sqrt(e) for e in Di_values]))
        D = np.matrix(np.diag(D))
    if code == 'mpmath':
        D12i = mp.matrix(mp.diag([mp.sqrt(e) for e in Di_values]))
        D = mp.matrix(mp.diag(D))
    S = D12i*A.transpose()*V12
    return S, D

    
def first_order_normal_form(H2, T=[], code='numpy', **kwargs):
    '''
    Perform linear calculations to map a given second-order Hamiltonian,
    expressed in canonical coordinates (q, p), to
    complex normal form coordinates xi, eta. Along the way, the symplectic linear map to
    real normal form is computed. The quantities xi and eta are defined as in my thesis,
    Chapter 1.
    
    Paramters
    ---------
    H2: dict
        Dictionary of the form H2 = {(i, j): h_ij}, so that the corresponding matrix 
        (h_ij) is a real, symmetric and positive definite (square) matrix. H2 can be sparsely defined;
        the mirrored indices will be added accordingly.
        The prime example is that H2 comes from extracting the second-order coefficients 
        in the Taylor expansion of a given Hamiltonian.
        
    T: matrix, optional
        Orthogonal matrix to change the ordering of canoncial
        coordinates and momenta, given here by default as (q1, ..., qn, p1, ..., pn), 
        into a different order. I.e. T transforms the 
        (n x n) block matrix
        
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
        into a matrix J' by matrix congruence: J' = T.transpose()*J*T.
             
    Returns
    -------
    dict
        Dictionary containing various linear maps. The entries of this dictionary are described in the following.
        
        S: The symplectic map diagonalizing H2 via S.transpose()*D*S = H2, where D is a diagonal matrix.
        Sinv: The inverse of S, i.e. the symplectic map to (real) normal form.
        H2: The input in matrix form.
        T: The (optional) matrix T described above.
        J: The (original) symplectic structure J' = J.transpose()*J*T, where J is the block-matrix given above.
        J2: The new symplectic structure for the (xi, eta)-coordinates.
        U: The unitary map from the S(p, q) = (u, v)-block coordinates to the (xi, eta)-coordinates.
        Uinv: The inverse of U.
        K: The linear map transforming H2 to complex normal form via K.transpose()*H2*K. 
            K is given by T.transpose()*Sinv*Uinv.
        Kinv: The inverse of K.
        rnf: The 'real' normal form, by which we understand the diagonalization of H2 relative to its 
            real normalizing coordinates (the matrix D described above).
        cnf: The 'complex' normal form, which is given as the representation of H2 in terms of the complex
            normalizing (xi, eta)-coordinates.
        D: The real entries of rnf in form of a list.
    '''
    # Create symmetric matrix from (sparse) dict
    dim = kwargs.get('dim', max(max(H2.keys())) + 1)
    assert dim%2 == 0, 'Dimension must be even.'
    
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        mp.mp.dps = kwargs.get('dps', 32) 
        sqrt2 = mp.sqrt(2)
    H2_matrix = column_matrix_2_code([[H2.get((i, j), 0) for i in range(dim)] for j in range(dim)], code=code)
    H2_matrix = 0.5*(H2_matrix + H2_matrix.transpose()) # the entries of the dictionary can be sparse and may contain zeros for mirrored values.
        
    # Now perform symplectic diagonalization
    if len(T) != 0: # transform H2 to default block ordering before entering williamson routine; the results will later be transformed back. This is easier instead of keeping track of orders inside the subroutines.
        H2_matrix = T*H2_matrix*T.transpose() 
    S, D = williamson(V=H2_matrix, code=code, **kwargs)
    
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
    # N.B. (p, Jq) -> (u, JSv) = (Sp, JSq) = (p, Jq) -> (Uu, JUv) = (xi, J eta). Thus:
    J2 = Uinv.transpose()*J*Uinv # the new symplectic structure with respect to the (xi, eta)-coordinates (holds also in the case len(T) != 0)
    Sinv = - J*S.transpose()*J
    K = Sinv*Uinv  # this map will transform to the new (xi, eta)-coordinates via K.transpose()*H2*K
    Kinv = U*S

    if len(T) != 0: # transform results back to the requested (q, p)-ordering
        S = T.transpose()*S*T
        Sinv = T.transpose()*Sinv*T
        D = T.transpose()*D*T
        J = T.transpose()*J*T
        
        K = T.transpose()*K
        Kinv = Kinv*T
    
    # assemble output
    out = {}
    out['S'] = S 
    out['Sinv'] = Sinv # this symplectic map will diagonalize H2 in its original 
    # (q, p)-coordinates via Sinv.transpose()*H2*Sinv. Sinv (and S) are symplectic wrt. J
    out['H2'] = H2_matrix # the matrix of second-order terms in H
    out['rnf'] = D # the diagonal matrix obtained as a result of the symplectic diagonalization of H2
    out['D'] = [D[i, i].real for i in range(len(D))]
    out['T'] = T
    out['J'] = J # the original symplectic structure
    out['J2'] = J2 # the new symplectic structure
    out['U'] = U # the unitary map from the S(p, q)=(u, v)-block coordinates to the (xi, eta)-coordinates
    out['Uinv'] = Uinv
    out['K'] = K
    out['Kinv'] = Kinv
    out['cnf'] = K.transpose()*H2_matrix*K # the representation of H2 in (xi, eta)-coordinates
    
    return out
    

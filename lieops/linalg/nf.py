# This file collects various algorithms to transform a given matrix into a 
# specific normal form -- or to decompose the matrix into a product of other matrices with various properties.

import mpmath as mp
import warnings
from sympy import Matrix as sympy_matrix
from scipy.linalg import polar, logm

from .checks import is_positive_definite
from .matrix import create_J, matrix_from_dict, create_pq2xieta

from lieops.linalg.congruence.takagi import symplectic_takagi, symplectic_takagi_old
from lieops.linalg.congruence.williamson import unitary_williamson, williamson
from lieops.linalg.similarity.symplectic import thm31

from njet.ad import getNargs
from njet.functions import get_package_name
from njet import derive    

def symlogs(X, **kwargs):
    r'''
    Let X be a complex symplectic matrix, i.e. a matrix satisfying
    X.transpose()@J@X = J.
    
    Then this routine will determine two matrices A and B so that
    X = exp(A)@exp(B),
    where A is an element in sp(n), the Lie-algebra of Sp(n) \subset Sp(2n; C) \cap U(2n),
    and B is an element of sp(2n; C), the Lie-algebra of Sp(2n; C).
    
    Parameters
    ----------
    X: ndarray
        An array representing the matrix X.
        
    **kwargs
        Optional keyworded arguments passed to 'lieops.linalg.similarity.symplectic.thm31' routine.
        
    Returns
    -------
    A: ndarray
        An array representing the matrix A.
        
    B: ndarray
        An array representing the matrix B.
    '''
    U, P = polar(X) # X = U@P with symplectic U and P.
    logP = logm(P) # logP.transpose()@J + J@logP = 0, i.e. logP is in Sp(n). Therefore we can diagonalize U symplectically:
    V = thm31(U, **kwargs) # V@U@V.transpose().conj() = D will be diagonal
    Vi = V.transpose().conj()
    D = V@U@Vi
    logD = logm(D) # logD.transpose()@J + J@logD = 0
    Y = Vi@logD@V
    return Y, logP

def normal_form(H2, T=[], mode='default', check: bool=False, **kwargs):
    r'''
    Perform linear calculations to transform a given second-order Hamiltonian,
    expressed in canonical coordinates (q, p), to
    complex normal form coordinates xi, eta. Along the way, the symplectic linear map to
    real normal form is computed. The quantities xi and eta are defined as in my thesis,
    Chapter 1.
    
    Paramters
    ---------
    H2:
        Symmetric matrix so that H2@J is diagonalizable.
        Attention: A check if H2@J is diagonalizable is not done explicitly.
        
    T: matrix, optional
        Permutation matrix to change the ordering of canoncial
        coordinates and momenta, given here by default as (q1, ..., qn, p1, ..., pn), 
        into a different order. I.e. T transforms the 
        (n x n) block matrix
        
             /  0   1  \
        J =  |         |
             \ -1   0  /
             
        into a matrix J' by matrix congruence: J' = T.transpose()@J@T.
        
    mode: str, optional
        Method of how to compute the symplectic matrices which conjugate-diagonalize H2.
        Supported modes are (default: 'default'):
        1) 'default' -- Use symplectic Takagi factorization
        2) 'classic' -- Use (unitary) Williamson diagonalization (works only if H2 is invertible.)
        
    check: boolean, optional
        Perform a consistency check whether H2@J is diagonalizable, using sympy.
             
    Returns
    -------
    dict
        Dictionary containing various linear maps. The items of this dictionary are described in the following.
        
        S      : The symplectic map diagonalizing H2 via S.transpose()@D@S = H2, where D is a diagonal matrix.
        Sinv   : The inverse of S, i.e. the symplectic map to (real) normal form.        
        H2     : The input in matrix form.
        T      : The (optional) matrix T described above.
        J      : The (original) symplectic structure J' = T.transpose()@J@T within which the input Hamiltonian was formulated.
                 Hereby J is the block-matrix from above.
        J2     : The new symplectic structure for the (xi, eta)-coordinates.
        U      : The unitary map from the S(q, p) = (u, v)-block coordinates to the (xi, eta)-coordinates.
        Uinv   : The inverse of U.
        K      : The linear map transforming (q, p) to (xi, eta)-coordinates. K is given by U@S@T.
        Kinv   : The inverse of K. Hence, it will transform H2 to complex normal form via Kinv.transpose()@H2@Kinv.
        rnf    : The 'real' normal form, by which we understand the diagonalization of H2 relative to the 
                 symplectic matrix S. Note that S might be complex if the Hesse matrix of H2 is not positive definite.
        cnf    : The 'complex' normal form, which is given as the representation of H2 in terms of the complex
                 normalizing (xi, eta)-coordinates (the 'new' complex symplectic structure).
        A      : A complex matrix transforming the underlying Hamiltonian H (whose Hesse-matrix corresponds to H2),
                 given in terms of complex (xi, eta)-coordinates, into normal form N via N = H o A.
                 It holds A = U@Sinv@Uinv. *Currently only returned for numpy input*
        S1, S2 : Elements of sp(2n; C) (the Lie-algebra of complex symplectic matrices) satisfying
                 A = exp(S1)@exp(S2). These matrices can be used to obtain respective polynomial representations
                 of the Lie-operator, mapping the given Hamiltonian H into its "first-order" normal form N (see
                 also the commment above). *Currently only returned for numpy input*
    ''' 
    dim = len(H2)
    assert dim%2 == 0, 'Dimension must be even.'
    code = get_package_name(H2)
        
    # Perform symplectic diagonalization
    if len(T) != 0: # transform H2 to default block ordering before entering williamson routine; the results will later be transformed back. This is easier instead of keeping track of orders inside the subroutines.
        T_ctr = T.transpose()
        H2 = T@H2@T_ctr

    J = create_J(dim//2)
    if code == 'mpmath':
        J = mp.matrix(J)
    
    if check:
        # consistency check: H2@J must be diagonalizable in order that the normal form can be computed.
        # Since computing the Jordan normal form is numerically unstable, we use sympy for this.
        # Note that this will only work for dim = 1 or dim = 2.
        J_symp = sympy_matrix(J)
        G_symp = sympy_matrix(H2)
        P_symp, JNF = (G_symp@J_symp).jordan_form()
        if not JNF.is_diagonal():
            raise RuntimeError('Jordan normal form of H2@J not diagonal.')
            
    if mode == 'new':
        # will become the default in future; TODO: some tests are failing
        S, X = symplectic_takagi(H2, **kwargs)
        S = S.transpose()
        Sinv = -J@S.transpose()@J
                
        D = Sinv.transpose()@H2@Sinv
        U = create_pq2xieta(dim=dim, code=code, **kwargs)
    elif mode == 'default':
        # will become OLD code. Using symplectic takagi assuming GJ is diagonalizable. This older version
        # works with eigenspaces and eigenvalues routine.
        # TODO: S may change the sign of the Hesse-matrix. Need to figure out the issue. 
        S, D, _ = symplectic_takagi_old(H2, check=check, **kwargs)
        S = S.transpose()
        Sinv = -J@S.transpose()@J
        U = create_pq2xieta(dim=dim, code=code, **kwargs)
    elif mode == 'classic':
        # OLD code, using Williamson or "unitary" Williamson.
        if is_positive_definite(H2):
            S, D = williamson(V=H2, **kwargs)

            U = create_pq2xieta(dim=dim, code=code, **kwargs)
            Sinv = -J@S.transpose()@J
        else:
            assert code == 'numpy' # TODO: mpmath support
            # apply new general routine in case H2 is not positive definite
            Sinv, D, U = unitary_williamson(M=H2, **kwargs) # U.conjugate()@Sinv.transpose()@H2@Sinv@U.conjugate().transpose() will be in (xi, eta)-canonical form.
            S = -J@Sinv.transpose()@J
    else:
        raise RuntimeError(f"Mode '{mode}' not recognized.")
    # The first dim columns of S denote (new) canonical positions u, the last dim columns of S
    # denote (new) canonical momenta v: S(q, p) = (u, v)
        
    # U is hermitian, therefore
    Uinv = U.transpose().conjugate()

    # N.B. (q, J*p) = (Sq, J*S*p) = (u, J*v) = (Uinv*U*u, J*Uinv*U*v) = (Uinv*xi, J*Uinv*eta) = (xi, Uinv.transpose().conjugate()@J@Uinv*eta).
    J2 = U@J@U.transpose() # the new symplectic structure (see my notes).
    #J2 = Uinv.transpose().conjugate()@J@Uinv # the new symplectic structure with respect to the (xi, eta)-coordinates (holds also in the case len(T) != 0)
    #   = U@J@U.transpose().conjuage() = [U@J@U.transpose()].conjugate()
    
    K = U@S # K(q, p) = (xi, eta)
    Kinv = Sinv@Uinv  # this map will transform to the new (xi, eta)-coordinates via Kinv.transpose()*H2*Kinv

    if len(T) != 0: # transform results back to the requested (q, p)-ordering
        S = T_ctr@S@T
        Sinv = T_ctr@Sinv@T
        J = T_ctr@J@T
        H2 = T_ctr@H2@T
        K = K@T
        Kinv = T_ctr@Kinv
    
    # assemble output
    out = {}
    out['S'] = S 
    out['Sinv'] = Sinv # this symplectic map will diagonalize H2 in its original 
    # (q, p)-coordinates via Sinv.transpose()*H2*Sinv. Sinv (and S) are symplectic wrt. J
    out['H2'] = H2 # the input matrix
    out['rnf'] = Sinv.transpose()@H2@Sinv # the diagonal matrix obtained as a result of the symplectic diagonalization of H2
    out['T'] = T
    out['J'] = J # the original symplectic structure
    out['J2'] = J2 # the new symplectic structure
    out['U'] = U # the unitary map from the S(p, q)=(u, v)-block coordinates to the (xi, eta)-coordinates
    out['Uinv'] = Uinv
    out['K'] = K # K(q, p) = (xi, eta)
    out['Kinv'] = Kinv
    out['cnf'] = Kinv.transpose()@H2@Kinv # the representation of H2 in (xi, eta)-coordinates
    out['D'] = D
    
    # Furthermore, compute the map A transforming the Hamiltonian (given in (xi, eta)-coordinates), 
    # whose Hessian corresponds to H2 above, into its first-order normal form.
    # For details see my notes (TODO: reference)
    if code != 'mpmath':
        out['A'] = U@Sinv@Uinv
        B1, B2 = symlogs(Sinv, tol2=kwargs.get('symlogs_tol2', 0))
        out['C1'] = U@B1@Uinv
        out['C2'] = U@B2@Uinv # so that A = exp(C1)@exp(C2)
        # Note that due to the nature of the matrix U, the Cj's are elements of sp(2n; C), so they admit a polynomial representation.
    return out

def first_order_nf_expansion(H, power: int=2, z=[], check: bool=False, n_args: int=0, tol: float=1e-14,
                             code='numpy', **kwargs):
    '''
    Return the Taylor-expansion of a Hamiltonian H in terms of first-order complex normal form coordinates
    around an optional point of interest. For the notation see my thesis.
    
    Parameters
    ----------
    H: callable
        A real-valued function of 2*n parameters (Hamiltonian).
        
    power: int, optional
        The maximal polynomial power of expansion. Must be >= 2.
    
    z: subscriptable, optional
        A point of interest around which we want to expand. If nothing specified,
        then the expansion will take place around zero.
        
    n_args: int, optional
        If H takes a single subscriptable as argument, define the number of arguments with this parameter.
        
    check: boolean, optional
        Turn on some basic checks:
        a) Warn if the expansion of the Hamiltonian around z contains gradient terms larger than a specific value. 
        b) Verify that the 2nd order terms in the expansion of the Hamiltonian agree with those from the linear theory.
        
    tol: float, optional
        An optional tolerance for checks.
        
    **kwargs
        Arguments passed to linalg.normal_form routine.
        
    Returns
    -------
    dict
        A dictionary of the Taylor coefficients of the Hamiltonian in normal form around z = (q, p), where the first n
        entries denote powers of xi, while the last n entries denote powers of eta.
        
    dict
        The output of linalg.normal_form routine, providing the linear map information at the requested point.
    '''
    assert power >= 2
    dim = getNargs(H, n_args=n_args)
    assert dim%2 == 0, 'Dimension must be even; try passing n_args argument.'
    
    # Step 1 (optional): Construct H locally around z (N.B. shifts are symplectic, as they drop out from derivatives.)
    # This step is required, because later on (at point (+)) we want to extract the Taylor coefficients, and
    # this works numerically only if we consider a function around zero.
    if len(z) > 0:
        assert len(z) == dim, f'Dimension ({len(z)}) of custom point mismatch (expected: {dim})'
        Hshift = lambda *x: H(*[x[k] + z[k] for k in range(len(z))])
    else:
        Hshift = H
        z = dim*[0]
    
    # Step 2: Obtain the Hesse-matrix of H.
    # N.B. we need to work with the Hesse-matrix here (and *not* with the Taylor-coefficients), because we want to get
    # a (linear) map K so that the Hesse-matrix of H o K is in CNF (complex normal form). This is guaranteed
    # if the Hesse-matrix of H is transformed to CNF.
    # Note that the Taylor-coefficients of H in 2nd-order are 1/2*Hesse_matrix. This means that at (++) (see below),
    # no factor of two is required.
    dHshift = derive(Hshift, order=2, n_args=dim)
    z0 = dim*[0]
    Hesse_dict = dHshift.hess(*z0)
    Hesse_matrix = matrix_from_dict(Hesse_dict, symmetry=1, n_rows=dim, n_cols=dim)
    if code == 'mpmath':
        Hesse_matrix = mp.matrix(Hesse_matrix)

    # Optional: Raise a warning in case the shifted Hamiltonian has first-order terms.
    if check:
        gradient = dHshift.grad() # the gradient of H is evaluated at z0 (note that H has been shifted to z0 above, so that z0 corresponds to the original point z).
        grad_vector = [gradient.get((k,), 0) for k in range(dim)]
        if any([abs(c) > tol for c in grad_vector]) > 0:
            warnings.warn(f'Input appears to have a non-zero gradient around the requested point\n{z}\nfor given tolerance {tol}:\n{grad_vector}')

    # Step 3: Compute the linear map to first-order complex normal form near z.
    nfdict = normal_form(Hesse_matrix, tol=tol, check=check, **kwargs)
    Kinv = nfdict['Kinv'] # Kinv.transpose()@Hesse_matrix@Kinv is in cnf; K(q, p) = (xi, eta)
    
    # Step 4: Obtain the expansion of the Hamiltonian up to the requested power.
    Kmap = lambda *zz: [sum([zz[k]*Kinv[j, k] for k in range(len(zz))]) for j in range(len(zz))] # TODO: implement column matrix class. Attention: Kinv[j, k] must stand on right-hand side, otherwise zz[k] may be inserted into a NumPy array!
    HK = lambda *zz: Hshift(*Kmap(*zz))
    dHK = derive(HK, order=power, n_args=dim)
    results = dHK(*z0, mult_drv=False) # mult_drv=False ensures that we obtain the Taylor-coefficients of the new Hamiltonian. (+)
    
    if check:
        # Check if the 2nd order Taylor coefficients of the derived shifted Hamiltonian agree in complex
        # normal form with the values predicted by linear theory.
        HK_hesse_dict = dHK.hess(tc=results)
        HK_hesse_dict = {k: v for k, v in HK_hesse_dict.items() if abs(v) > tol}
        for k in HK_hesse_dict.keys():
            diff = abs(HK_hesse_dict[k] - nfdict['cnf'][k[0], k[1]]) # (++)
            if diff > tol:
                raise RuntimeError(f'CNF entry {k} does not agree with Hamiltonian expansion: diff {diff} > {tol} (tol).')
        
    return results, nfdict

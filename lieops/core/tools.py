# collection of specialized tools operating on polynomials

import numpy as np
import lieops.core.lie
from lieops.linalg.matrix import create_J

def poly2ad(pin):
    '''
    Compute a (2n)x(2n)-matrix representation of a homogenous second-order polynomial A,
    so that if z_j denotes the projections onto the canonical coordinate components, then
    
    {A, z_j} = A_{ij} z_i                    (1)
    
    holds. The brackets { , } denote the poisson bracket. The values A_{ij} will be determined.
    
    Remark: The order of the indices in Eq. (1) has been chosen to guarantee that matrix multiplication
            and adjoint both maintain the same order:
            {A, {B, z_j}} = (A o B)_{ij} z_i
    
    Parameters
    ----------
    pin: poly
        The polynomial to be converted.
        
    Returns
    -------
    array-like
        A complex matrix corresponding to the representation.
    '''
    assert pin.maxdeg() == 2 and pin.mindeg() == 2
    dim = pin.dim
    dim2 = dim*2
    poisson_factor = pin._poisson_factor
    A = np.zeros([dim2, dim2], dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            mixed_key = [0]*dim2 # key belonging to xi_i*eta_j
            mixed_key[i] += 1
            mixed_key[j + dim] += 1
            A[i, j] = pin.get(tuple(mixed_key), 0)*-poisson_factor
            A[j + dim, i + dim] = pin.get(tuple(mixed_key), 0)*poisson_factor
            
            if i != j: # if i and j are different, than the key in the polynomial already
                # corresponds to the sum of the ij and the ji-coefficient. But if they are equal,
                # then the values has to be multiplied by 2, because we have to use the ij + ji-components.
                ff = 1
            else:
                ff = 2
                
            hom_key_xi = [0]*dim2 # key belonging to xi_i*xi_j
            hom_key_xi[i] += 1
            hom_key_xi[j] += 1
            A[i, j + dim] = pin.get(tuple(hom_key_xi), 0)*poisson_factor*ff

            hom_key_eta = [0]*dim2 # key belonging to eta_i*eta_j
            hom_key_eta[i + dim] += 1
            hom_key_eta[j + dim] += 1
            A[i + dim, j] = pin.get(tuple(hom_key_eta), 0)*-poisson_factor*ff
    return A

def ad2poly(A, tol=0, poisson_factor=-1j, **kwargs):
    '''
    Transform a complex (2n)x(2n)-matrix representation of a polynomial back to 
    its polynomial xi/eta-representation. This is the inverse of the 'poly2ad' routine.
    
    Attention: A matrix admits a polynomial representation if and only if it is an element
               of sp(2n; C), the Lie-algebra of symplectic complex (2n)x(2n)-matrices. By default
               this routine will *not* check against this property (but see the information below).
    
    Parameters
    ----------
    A: array-like
        Matrix representing the polynomial.
        
    tol: float, optional
        A tolerance to check a sufficient property of the given matrix to be a valid representation.
        No check will be made if 'tol' is zero (default).
        
    poisson_factor: complex, optional
        A factor defining the poisson structure of the sought polynomial. By default this factor
        is -1j, corresponding to the poisson structure of the complex xi/eta coordinates.
        
    **kwargs
        Optional keyworded arguments passed to lieops.core.lie.poly routine.
        
    Returns
    -------
    poly
        Polynomial corresponding to the matrix.
    '''
    assert A.shape[0] == A.shape[1]
    dim2 = A.shape[0]
    assert dim2%2 == 0
    dim = dim2//2
    
    values = {}
    for i in range(dim):
        for j in range(dim):
            
            if tol > 0:
                # Check if the given matrix is an element of sp(2n; C). If this check fails,
                # no valid polynomial representation can be obtained.
                error_msg = f'{A}\nThe given matrix does not appear to be an adjoint representation of a polynomial, using a check with tolerance {tol} '
                c1 = abs(A[i, j] + A[j + dim, i + dim])
                c2 = abs(A[i, j + dim] - A[j, i + dim])
                c3 = abs(A[i + dim, j] - A[j + dim, i])
                assert c1 < tol, error_msg + f'({c1}).'
                assert c2 < tol, error_msg + f'({c2}).'
                assert c3 < tol, error_msg + f'({c3}).'
            
            mixed_key = [0]*dim2 # key belonging to a coefficient of mixed xi/eta variables.
            mixed_key[i] += 1
            mixed_key[j + dim] += 1            
            values[tuple(mixed_key)] = A[i, j]*-1/poisson_factor
            
            # The factor 'ff' comes from the fact that the poly objects terms of the form xi_i*xi_j (for i != j) and xi_j*xi_i are equal.
            if i != j:
                ff = 1
            else:
                ff = 2
            
            hom_key_xi = [0]*dim2 # key belonging to a coefficient xi-xi variables.
            hom_key_xi[i] += 1
            hom_key_xi[j] += 1
            values[tuple(hom_key_xi)] = A[i, j + dim]/ff/poisson_factor
            
            hom_key_eta = [0]*dim2 # key belonging to a coefficient eta-eta variables.
            hom_key_eta[i + dim] += 1
            hom_key_eta[j + dim] += 1
            values[tuple(hom_key_eta)] = A[i + dim, j]/ff*-1/poisson_factor
    return lieops.core.lie.poly(values=values, dim=dim, poisson_factor=poisson_factor, **kwargs)

def poly2vec(p):
    '''
    Map a first-order polynomial to its respective vector in matrix representation 
    (see also 'poly2ad' routine)
    '''
    assert p.maxdeg() == 1 and p.mindeg() == 1
    dim = p.dim
    out = np.zeros(dim*2, dtype=np.complex128)
    for k, v in p.items():
        j = list(k).index(1)
        out[j] = v
    return out

def vec2poly(v):
    '''
    The inverse of 'poly2vec' routine.
    '''
    dim2 = len(v)
    assert dim2%2 == 0, 'Dimension must be even.'
    xieta = lieops.core.lie.create_coords(dim2//2)
    return sum([xieta[k]*v[k] for k in range(dim2)])

def poly3ad(pin):
    '''
    Compute a (2n + 1)x(2n + 1)-matrix representation of a second-order polynomial (without
    constant term), given in terms of complex xi/eta coordinates, 
    so that if z_j denote the basis vectors, then:
    
    {p, z_j} = p_{ij} z_i + r_j
    
    holds. The brackets { , } denote the poisson bracket. The values p_{ij} and r_j will be determined.
    
    Parameters
    ----------
    pin: poly
        The polynomial to be converted.
        
    Returns
    -------
    array-like
        A complex matrix corresponding to the representation.
    '''
    assert pin.maxdeg() <= 2 and pin.mindeg() >= 1 # Regarding the second condition: Constants have no effect as 'ad' and therefore 'ad' can not be inverted. Since we want poly3ad to be invertible, we have to restrict to polynomials without such constant terms.
    dim = pin.dim
    dim2 = dim*2
    poisson_factor = pin._poisson_factor
    # extended space: (xi/eta)-phase space + constants.
    pmat = np.zeros([dim2 + 1, dim2 + 1], dtype=np.complex128) 
    # 1. Add the representation with respect to 2x2-matrices:
    pin2 = pin.homogeneous_part(2)
    if len(pin2) != 0:
        pmat[:dim2, :dim2] = poly2ad(pin2)
    # 2. Add the representation with respect to the scalar:
    pin1 = pin.homogeneous_part(1)
    if len(pin1) != 0:
        for k in range(dim):
            xi_key = [0]*dim2
            xi_key[k] = 1
            pmat[dim2, k + dim] = pin1.get(tuple(xi_key), 0)*poisson_factor

            eta_key = [0]*dim2
            eta_key[k + dim] = 1
            pmat[dim2, k] = pin1.get(tuple(eta_key), 0)*-poisson_factor
    return pmat

def ad3poly(A, **kwargs):
    '''
    The inverse of the 'poly3ad' routine.
    '''
    assert A.shape[0] == A.shape[1]
    dim2 = A.shape[0] - 1
    assert dim2%2 == 0
    dim = dim2//2
    # 1. Get the 2nd-order polynomial associated to the dim2xdim2 submatrix:
    p2 = ad2poly(A[:dim2, :dim2], **kwargs)
    poisson_factor = p2._poisson_factor
    if len(p2) == 0:
        p2 = 0
    # 2. Get the first-order polynomials associated to the remaining line:
    xieta = lieops.core.lie.create_coords(dim)
    for k in range(dim):
        eta_k_coeff = A[dim2, k]/-poisson_factor
        xi_k_coeff = A[dim2, k + dim]/poisson_factor
        p2 += xieta[k]*xi_k_coeff
        p2 += xieta[k + dim]*eta_k_coeff
    return p2

def poly3vec(p, **kwargs):
    '''
    Map a Lie polynomial of degree <= 1 to its respective (2n + 1)-dimensional
    vector. The last dimension is hereby dedicated to the constants. 
    '''
    assert p.maxdeg() <= 1
    p0 = p.homogeneous_part(0)
    p1 = p.homogeneous_part(1)
    vec1 = poly2vec(p1, **kwargs)
    return np.append(vec1, p0.get((0,)*p.dim*2, 0))

def vec3poly(vec, **kwargs):
    '''
    The inverse of the poly3vec routine.
    '''
    assert len(vec)%2 == 1
    n2 = len(vec) - 1
    p1 = vec2poly(vec[:-1], **kwargs)
    p1[(0,)*n2] = vec[-1]
    return p1

def const2poly(*const, poisson_factor=-1j, **kwargs):
    '''
    Let c_1, ..., c_{2n} be constants. Then compute a first-order polynomial g_1
    so that {g_1, z_j} = z_j + c_j holds.
    '''
    dim2 = len(const)
    assert dim2%2 == 0, 'Dimension must be even.'
    dim = dim2//2
    J = create_J(dim)
    xieta = lieops.core.lie.create_coords(dim, poisson_factor=poisson_factor, **kwargs)
    return sum([xieta[k]*(J@const)[k] for k in range(dim2)])/poisson_factor

def poly2const(p1):
    '''
    The inverse of the 'const2poly' routine.
    '''
    assert p1.mindeg() == 1 and p1.maxdeg() == 1
    xieta = lieops.core.lie.create_coords(p1.dim, poisson_factor=p1._poisson_factor, max_power=p1.max_power)
    return [(p1@xe)[(0,)*p1.dim*2] for xe in xieta] # "p1@xe - xe" not necessary, we access the constant immediately

def action_on_poly(*mu, C, func=lambda z: z):
    '''
    The poisson bracket of the polynomial X := sum_k mu_k epsilon_k 
    (notation as in Ref. [1], where epsilon_k denotes the k-th action and mu_k 
    the k-th tune), applied to some arbitrary polynomial C, can be computed directly 
    due to the fact that the xi/eta-coordinates constitute an "eigenbasis" of X.
    
    This routine will perform the calculation and return the resulting polynomial.
    
    Parameters
    ----------
    mu: float (or complex)
        The tunes of the action-operator
    
    C: poly
        The polynomial on which the action operator should be applied.
        
    func: callable (optional)
        A function f so that f(X) is applied instead.
        
    Returns
    -------
    poly
        The poly object {X, C} (or {f(X), C}, if a function has been provided).
    '''
    assert len(mu) == C.dim
    return lieops.core.lie.poly(values={powers: v*func(sum([(powers[k] - powers[k + C.dim])*mu[k] for k in range(C.dim)])*1j) for powers, v in C.items()}, 
                dim=C.dim, max_power=C.max_power)
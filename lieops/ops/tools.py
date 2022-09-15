# collection of specialized tools operating on polynomials

import numpy as np
from scipy.linalg import expm

import lieops.ops.lie
from lieops.linalg.matrix import adjoint, vecmat, matvec


def poly2ad(pin):
    '''
    Compute a (2n)x(2n)-matrix representation of a homogenous second-order polynomial, given
    in terms of complex xi/eta coordinates, so that if z_j denote the basis vectors, then:
    
    {p, z_j} = p_{ij} z_i
    
    holds. The brackets { , } denote the poisson bracket. The values p_{ij} will be determined.
    
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
    pmat = np.zeros([dim2, dim2], dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            mixed_key = [0]*dim2 # key belonging to xi_i*eta_j
            mixed_key[i] += 1
            mixed_key[j + dim] += 1
            pmat[i, j] = pin.get(tuple(mixed_key), 0)*-poisson_factor
            pmat[j + dim, i + dim] = pin.get(tuple(mixed_key), 0)*poisson_factor
            
            if i != j: # if i and j are different, than the key in the polynomial already
                # corresponds to the sum of the ij and the ji-coefficient. But if they are equal,
                # then the values has to be multiplied by 2, because we have to use the ij + ji-components.
                ff = 1
            else:
                ff = 2
                
            hom_key_xi = [0]*dim2 # key belonging to xi_i*xi_j
            hom_key_xi[i] += 1
            hom_key_xi[j] += 1
            pmat[i, j + dim] = pin.get(tuple(hom_key_xi), 0)*poisson_factor*ff

            hom_key_eta = [0]*dim2 # key belonging to eta_i*eta_j
            hom_key_eta[i + dim] += 1
            hom_key_eta[j + dim] += 1
            pmat[i + dim, j] = pin.get(tuple(hom_key_eta), 0)*-poisson_factor*ff
    return pmat

def ad2poly(amat, tol=0, poisson_factor=-1j):
    '''
    Transform a complex (2n)x(2n)-matrix representation of a polynomial back to 
    its polynomial xi/eta-representation. This is the inverse of the 'poly2ad' routine.
    
    Parameters
    ----------
    amat: array-like
        Matrix representing the polynomial.
        
    tol: float, optional
        A tolerance to check a sufficient property of the given matrix to be a valid representation.
        No check will be made if 'tol' is zero (default).
        
    poisson_factor: complex, optional
        A factor defining the poisson structure of the sought polynomial. By default this factor
        is -1j, corresponding to the poisson structure of the complex xi/eta coordinates.
        
    Returns
    -------
    poly
        Polynomial corresponding to the matrix.
    '''
    assert amat.shape[0] == amat.shape[1]
    dim2 = amat.shape[0]
    assert dim2%2 == 0
    dim = dim2//2
    
    values = {}
    for i in range(dim):
        for j in range(dim):
            
            if tol > 0:
                # If the following check fails, amat is not an adjoint representation.
                # This condition can be derived by considering a general second-order polynomial Q
                # which has to satisfy:
                # {Q, xi_k} = amat[i, k]*xi_i + amat[i + dim, k]*eta_i
                # and
                # {Q, eta_k} = amat[i, dim + k]*xi_i + amat[i + dim, k + dim]*eta_i
                # (where repeated indices are summed up).
                err = abs(amat[i, j] + amat[j + dim, i + dim])
                assert err < tol, f'The given matrix does not appear to be an adjoint representation of a polynomial: {err} >= {tol} (tol).'
            
            mixed_key = [0]*dim2 # key belonging to a coefficient of mixed xi/eta variables.
            mixed_key[i] += 1
            mixed_key[j + dim] += 1            
            values[tuple(mixed_key)] = amat[i, j]*-1/poisson_factor
            
            # The factor 'ff' comes from the fact that the poly objects terms of the form xi_i*xi_j (for i != j) and xi_j*xi_i are equal.
            if i != j:
                ff = 1
            else:
                ff = 2
            
            hom_key_xi = [0]*dim2 # key belonging to a coefficient xi-xi variables.
            hom_key_xi[i] += 1
            hom_key_xi[j] += 1
            if tol > 0:
                assert abs(amat[i, j + dim] - amat[j, i + dim]) < tol # consistency check
            values[tuple(hom_key_xi)] = amat[i, j + dim]/ff/poisson_factor
            
            hom_key_eta = [0]*dim2 # key belonging to a coefficient eta-eta variables.
            hom_key_eta[i + dim] += 1
            hom_key_eta[j + dim] += 1
            if tol > 0:
                assert abs(amat[i + dim, j] - amat[j + dim, i]) < tol # consistency check
            values[tuple(hom_key_eta)] = amat[i + dim, j]/ff*-1/poisson_factor
    return lieops.ops.lie.poly(values=values, poisson_factor=poisson_factor)

def poly1repr(p):
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

def repr1poly(v):
    '''
    The inverse of 'poly1repr' routine.
    '''
    dim2 = len(v)
    assert dim2%2 == 0, 'Dimension must be even.'
    xieta = lieops.ops.lie.create_coords(dim2//2)
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
    assert pin.maxdeg() <= 2 and pin.mindeg() >= 1 # To the second condition: Constants have zero-effect as 'ad' and therefore can not yield an invertible map. Since we want poly3ad to be invertible, we have to restrict to polynomials without constant terms.
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

def ad3poly(amat, **kwargs):
    '''
    The inverse of the 'poly3ad' routine.
    '''
    assert amat.shape[0] == amat.shape[1]
    dim2 = amat.shape[0] - 1
    assert dim2%2 == 0
    dim = dim2//2
    # 1. Get the 2nd-order polynomial associated to the dim2xdim2 submatrix:
    p2 = ad2poly(amat[:dim2, :dim2], **kwargs)
    poisson_factor = p2._poisson_factor
    if len(p2) == 0:
        p2 = 0
    # 2. Get the first-order polynomials associated to the remaining line:
    xieta = lieops.ops.lie.create_coords(dim)
    for k in range(dim):
        eta_k_coeff = amat[dim2, k]/-poisson_factor
        xi_k_coeff = amat[dim2, k + dim]/poisson_factor
        p2 += xieta[k]*xi_k_coeff
        p2 += xieta[k + dim]*eta_k_coeff
    return p2

def get_2flow(ham, tol=1e-12):
    '''
    Compute the exact flow of a 2nd-order Hamiltonian, for polynomials up to second-order.
    I.e. compute the solution of
        dz/dt = {H, z}, z(0) = p,
    where { , } denotes the poisson bracket, H the requested Hamiltonian.
    Hereby p must be a polynomial of order <= 2.
    
    Parameters
    ----------
    ham: poly
        A polynomial of order <= 2.
        
    tol: float, optional
        A tolerance to check whether the adjoint matrix of the matrix-representation of the given Hamiltonian
        admits an invertible matrix of eigenvalues according to np.linalg.eig. In this case, one can use
        fast matrix multiplication in the resulting flow. Otherwise we have to rely on scipy.linalg.expm.
    '''
    poisson_factor = ham._poisson_factor
    
    Hmat = poly3ad(ham) # Hmat: (2n + 1)x(2n + 1)-matrix
    adHmat = adjoint(Hmat) # adHmat: (m**2)x(m**2)-matrix; m := 2n + 1
    
    # Alternative:
    evals, M = np.linalg.eig(adHmat)
    check = abs(np.linalg.det(M)) < tol
    if check:
        # in this case we have to rely on a different method to calculate the matrix exponential.
        # for the time being we shall use scipy's expm routine.
        expH = expm(adHmat)
    else:
        Mi = np.linalg.inv(M) # so that M@np.diag(evals)@Mi = adHmat holds.
        # compute the exponential exp(t*adHmat) = exp(M@(t*D)@Mi) = M@exp(t*D)@Mi:
        expH = M@np.diag(np.exp(evals))@Mi
    
    # Let Y be a (m**2)-vector (or (m**2)x(m**2)-matrix) and @ the composition
    # with respect to the (m**2)-dimensional space. Then
    # d/dt (exp(t*adHmat)@Y) = adHmat@exp(t*adHmat)@Y, so that
    # Z := exp(t*adHmat)@Y solves the differential equation
    # dZ/dt = adHmat@Z with Z(0) = Y.
    #
    # In the case that Y was a vector (and so Z), then we can write Z = vecmat(z) for
    # a suitable (m)x(m)-matrix z.
    # By exchanging differentiation d/dt and vecmat we then obtain:
    # vecmat(dz/dt) = adjoint(Hmat)@vecmat(z) = vecmat(Hmat@z - z@Hmat),
    # Consequently:
    # dz/dt = Hmat@z - z@Hmat = [Hmat, z],
    # where the [ , ] denotes the commutator of matrices.
    # Hereby vectmat(y) = Y = Z(0) = vectmat(z(0)), i.e. y = z(0) for the respective
    # start conditions, with (m)x(m)-matrix y.
    #
    # Using this notation, we define the flow function as follows:
    def flow(p, t=1, **kwargs):
        '''
        Compute the solution z so that
        dz/dt = {H, z}, z(0) = p,
        where { , } denotes the poisson bracket, H the requested Hamiltonian.
        Hereby p must be a polynomial of order <= 2.
        
        The solution thus corresponds to
        z(t) = exp(t:H:)p

        Parameters
        ----------
        p: poly
            The start polynomial of order <= 2.
            
        t: float, optional
            An optional parameter to control the flow (see above).
        '''
        if not isinstance(p, lieops.ops.lie.poly):
            return p
        
        assert poisson_factor == p._poisson_factor, 'Hamiltonian and given polynomial are instantiated with respect to different poisson structures.'
        
        if t != 1:
            if check:
                expH_t = expm(t*adHmat)
            else:
                expH_t = M@np.diag(np.exp(t*evals))@Mi                
        else:
            expH_t = expH
        p0 = p.homogeneous_part(0) # the constants will be reproduced in the end (by the '1' in the flow)
        p1 = p.extract(key_cond=lambda x: sum(x) >= 1)
        result = p0
        if len(p1) > 0:
            Y = vecmat(poly3ad(p1))
            Z = expH_t@Y
            result += ad3poly(matvec(Z), poisson_factor=poisson_factor)
        return result
    return flow
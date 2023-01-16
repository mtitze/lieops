import numpy as np
from scipy.linalg import expm
import warnings

from lieops.core.tools import poly3ad, ad3poly, vec3poly, poly3vec
import lieops.core.lie

def get_2flow(ham, tol=1e-12):
    '''
    Compute the exact flow of a Hamiltonian, modeled by a polynomial of first or second-order.
    I.e. compute the solution of
        dz/dt = {H, z}, z(0) = p,
    where { , } denotes the poisson bracket, H the requested Hamiltonian.
    Hereby H and p must be polynomials of order <= 2.
    
    Parameters
    ----------
    ham: poly
        A polynomial of order <= 2.
        
    tol: float, optional
        A tolerance to check whether the matrix-representation of the given Hamiltonian
        admits an invertible matrix of eigenvalues according to np.linalg.eig. In this case, one can use
        fast matrix multiplication in the resulting flow. Otherwise we have to rely on scipy.linalg.expm.
    '''
    if ham.maxdeg() == 0:
        # return the identity map
        def flow(p, t=1, **kwargs):
            return p
        return flow
    
    assert ham.maxdeg() <= 2, 'Hamiltonian of degree <= 2 required.'
    poisson_factor = ham._poisson_factor
    
    Hmat = poly3ad(ham) # Hmat: (2n + 1)x(2n + 1)-matrix
    
    # Alternative:
    evals, M = np.linalg.eig(Hmat)
    check = abs(np.linalg.det(M)) < tol
    if check:
        # in this case we have to rely on a different method to calculate the matrix exponential.
        # for the time being we shall use scipy's expm routine.
        expH = expm(Hmat)
    else:
        Mi = np.linalg.inv(M) # so that M@np.diag(evals)@Mi = Hmat holds.
        # compute the exponential exp(t*Hmat) = exp(M@(t*D)@Mi) = M@exp(t*D)@Mi:
        expH = M@np.diag(np.exp(evals))@Mi

    def flow(p, t=1, **kwargs):
        '''
        Compute the solution z so that
        dz/dt = {H, z}, z(0) = p,
        where { , } denotes the poisson bracket, H the requested Hamiltonian.
        
        The solution thus corresponds to
        z(t) = exp(t:H:)p

        Parameters
        ----------
        p: poly
            The start polynomial.
            
        t: float, optional
            An optional parameter to control the flow (see above).
        '''
        if not isinstance(p, lieops.core.lie.poly):
            warnings.warn('lieops.core.lie.poly input expected')
            return p
        
        assert poisson_factor == p._poisson_factor, 'Hamiltonian and given polynomial are instantiated with respect to different poisson structures.'
        
        if t != 1:
            if check:
                expH_t = expm(Hmat*t)
            else:
                expH_t = M@np.diag(np.exp(evals*t))@Mi  
        else:
            expH_t = expH
            
        maxdeg = p.maxdeg()
        p0 = p.homogeneous_part(0) # the constants will be reproduced in the end (by the '1' in the flow)
        result = p0
        if maxdeg > 0:
            p1 = p.homogeneous_part(1)
            Y = poly3vec(p1)
            Z = expH_t@Y
            result += vec3poly(Z)
        if maxdeg > 1:
            p_rest = p.extract(key_cond=lambda x: sum(x) > 1)
            # compute the flow using the Pull-back property of exp(:f:)
            dim = p.dim
            dim2 = dim*2
            xieta = lieops.core.lie.create_coords(dim=dim)
            unit_vectors = np.concatenate([np.eye(dim2), np.zeros([1, dim2])], axis=0)
            Z = expH_t@unit_vectors
            xietaf = [vec3poly(zz) for zz in Z.transpose()]
            result += p_rest(*xietaf)
        return result
    return flow

import numpy as np
from scipy.linalg import expm

from lieops.linalg.matrix import adjoint, vecmat, matvec
from lieops.ops.tools import poly3ad, ad3poly
import lieops.ops.lie

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

import numpy as np
from scipy.linalg import expm
import warnings

from lieops.core.tools import poly3ad, ad3poly, vec3poly, poly3vec
from lieops.linalg.common import ndsupport
from lieops.linalg.matrix import emat

import lieops.core.lie

np_eig_nd = ndsupport(np.linalg.eig, n_out_args=2)
np_inv_nd = ndsupport(np.linalg.inv)

_checkf = ndsupport(lambda x: abs(np.linalg.det(x)))
expm_nd = ndsupport(expm)
_diagexp = ndsupport(lambda d: np.diag(np.exp(d)), n_inp_axes=1)

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
    
    evals, M = np_eig_nd(Hmat) # Compute the eigenvalues and eigenvectors of Hmat
    check = (_checkf(M) < tol).any()
    if check:
        # in this case we have to rely on a different method to calculate the matrix exponential.
        # for the time being we shall use scipy's expm routine.
        expH = emat(expm_nd(Hmat))
    else:
        #Mi = np.linalg.inv(M) so that M@np.diag(evals)@Mi = Hmat holds.
        Mi = np_inv_nd(M)
        # compute the exponential exp(t*Hmat) = exp(M@(t*D)@Mi) = M@exp(t*D)@Mi:
        M, Mi = emat(M), emat(Mi)
        expH = M@emat(_diagexp(evals))@Mi

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
                expH_t = emat(expm_nd(Hmat*t))
            else:
                expH_t = M@emat(_diagexp(evals*t))@Mi
        else:
            expH_t = expH
                        
        maxdeg = p.maxdeg()
        p0 = p.homogeneous_part(0) # the constants will be reproduced in the end (by the '1' in the flow)
        result = p0
        
        if maxdeg > 0:
            p1 = p.homogeneous_part(1)
            Y = emat(poly3vec(p1))
            Z = expH_t@Y
            result += vec3poly(Z.matrix) # Z (and Y) are modeled here by 'emat' objects (extended matrices). This object also works for vectors, which is the case here.
            
        if maxdeg > 1: # compute the flow using the Pull-back property of exp(:f:)
            p_rest = p.extract(key_cond=lambda x: sum(x) > 1)
            dim = p.dim
            dim2 = dim*2
            
            # Create unit vectors where each has attached one zero in the end (for the dim2 + 1 scenario).
            shape = expH_t.shape[2:]
            zeros = np.zeros(shape, dtype=np.complex128)
            ones = np.ones(shape, dtype=np.complex128)
            E = np.array([[zeros if i != k else ones for i in range(dim2)] for k in range(dim2 + 1)])
            
            # Apply the matrix representation to the unit vectors
            Z = (expH_t@E).transpose()
            xietaf = [vec3poly(Z.matrix[j]) for j in range(len(Z))]
            result += p_rest(*xietaf)
            
        return result
    
    return flow

# collection of specialized tools operating on polynomials

import numpy as np
from .lie import poly

def poly2ad(pin):
    '''
    Compute a (2n)x(2n)-matrix representation of a homogenous second-order polynomial.
    
    Parameters
    ----------
    pin: poly
        The polynomial to be converted.
        
    Returns
    -------
    array-like
        A matrix corresponding to the representation.
    '''
    assert pin.maxdeg() == 2 and pin.mindeg() == 2
    dim = pin.dim
    dim2 = dim*2
    pmat = np.zeros([dim2, dim2], dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            mixed_key = [0]*dim2 # key belonging to xi_i*eta_j
            mixed_key[i] += 1
            mixed_key[j + dim] += 1
            pmat[i, j] = pin.get(tuple(mixed_key), 0)*1j
            pmat[j + dim, i + dim] = pin.get(tuple(mixed_key), 0)*-1j
            
            if i != j: # if i and j are different, than the key in the polynomial already
                # corresponds to the sum of the ij and the ji-coefficient. But if they are equal,
                # then the values has to be multiplied by 2, because we have to use the ij + ji-components.
                ff = 1
            else:
                ff = 2
                
            hom_key_xi = [0]*dim2 # key belonging to xi_i*xi_j
            hom_key_xi[i] += 1
            hom_key_xi[j] += 1
            pmat[i, j + dim] = pin.get(tuple(hom_key_xi), 0)*-1j*ff

            hom_key_eta = [0]*dim2 # key belonging to eta_i*eta_j
            hom_key_eta[i + dim] += 1
            hom_key_eta[j + dim] += 1
            pmat[i + dim, j] = pin.get(tuple(hom_key_eta), 0)*1j*ff
    return pmat

def ad2poly(amat, tol=0):
    '''
    Transform a (2n)x(2n)-matrix representation of a polynomial back to its polynomial object.
    
    Parameters
    ----------
    amat: array-like
        Matrix representing the polynomial.
        
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
            mixed_key = [0]*dim2 # key belonging to a coefficient of mixed xi/eta variables.
            mixed_key[i] += 1
            mixed_key[j + dim] += 1            
            values[tuple(mixed_key)] = amat[i, j]*-1j
            
            if i != j:
                ff = 1
            else:
                ff = 2
            
            hom_key_xi = [0]*dim2 # key belonging to a coefficient xi-xi variables.
            hom_key_xi[i] += 1
            hom_key_xi[j] += 1
            if tol > 0:
                assert abs(amat[i, j + dim] - amat[j, i + dim]) < tol # consistency check; if this fails, amat is not a representation
            values[tuple(hom_key_xi)] = amat[i, j + dim]*1j/ff
            
            hom_key_eta = [0]*dim2 # key belonging to a coefficient eta-eta variables.
            hom_key_eta[i + dim] += 1
            hom_key_eta[j + dim] += 1
            if tol > 0:
                assert abs(amat[i + dim, j] - amat[j + dim, i]) < tol # consistency check; if this fails, amat is not a representation
            values[tuple(hom_key_eta)] = amat[i + dim, j]*-1j/ff
    return poly(values=values)


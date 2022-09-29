# This file contains routines which are not immediately related to matrix operations, but may be used
# in several places in algorithms involving lieops.linalg routines.
import numpy as np

def identifyPairs(D, condition, **kwargs):
    '''
    Let D = [a, b, ..., c, ..., d, ...] be a list of complex values.
    This routine will identify the pairs (a, c), (b, d), ... and return
    their indices, according to a given condition.
    
    Parameters
    ----------
    D: list
        A list of even length containing the pairs of values.
        
    condition: callable
        A function f taking two values and returning a boolean, so that we will
        return the indices where f(a, b) is true.
    '''
    dim2 = len(D)
    assert dim2%2 == 0
    dim = dim2//2

    error_msg = f'Error identifying pairs. Check input or condition.'

    # Step 1: Determine the pairs on the diagonal which should be mapped.
    ind1, ind2 = [], []
    for i in range(len(D)):
        if i in ind1 or i in ind2:
            continue
        for j in range(len(D)):
            if j in ind1 or j in ind2 or j == i:
                continue
            if condition(D[i], D[j]):
                ind1.append(i)
                ind2.append(j)
                break # index i consumed
                
    assert len(ind1) == len(ind2), error_msg
    return list(zip(ind1, ind2))

def get_orientation(Dref, D, tol=1e-10):
    '''
    For two lists of values Dref = [a, b, ..., -a, ..., -b, ...], D = [c, d, ..., -c, ..., -d, ...], this routine
    will attempt to identify the indices of those values from list Dref with the one of D in such a way that
    whenever one value can be transformed by multiplication by +/-i to another, the indices will be determined.
    
    Parameters
    ----------
    Dref: list
        The 'reference' list
        
    D: list
        The second list
        
    tol: float, optional
        Tolerance by which we identify two values to be equal.
        
    Returns
    -------
    np.array
        A numpy array whose (i, j) entry denotes the factor f so that D[j]*f = Dref[i] holds, where f in [i, -i].
    '''
    dim2 = len(D)
    assert len(Dref) == dim2
    orient = np.zeros([dim2, dim2], dtype=np.complex128)
    check = 0 
    for k in range(dim2):
        a = Dref[k]
        for l in range(dim2):
            b = D[l]
            if abs(a - 1j*b) < tol:
                orient[k, l] = 1j
            elif abs(a + 1j*b) < tol:
                orient[k, l] = -1j
            else:
                check += 1
                continue
    # check if there are exactly 2 entries for each column. If this is not the case, something went wrong.
    assert check == dim2**2 - 2*dim2, f'Orientation could not be determined using tol: {tol}'
    return orient

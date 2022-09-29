import numpy as np
import cmath

def rotate_2block(x):
    r'''
    Helper function to "rotate" a 2x2-matrix M of the form
    
        / 0    x \
    M = |        |
        \ -x   0 /
        
    by means of a unitary matrix U so that U.transpose()@M@U has the same form as M, but
    in which x has no imaginary part and is >= 0.
    
    Parameters
    ----------
    x:
        The entry in the top right corner of M.
    
    Returns
    -------
    U:
        Unitary 2x2 matrix with the property as described above.
    '''
    phi = -cmath.phase(x)
    return np.array([[np.exp(1j*phi/2), 0], [0, np.exp(1j*phi/2)]])
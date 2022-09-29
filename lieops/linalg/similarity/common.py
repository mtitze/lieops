# This file contains common routines which may be used in more special routines in this folder.

import numpy as np
import mpmath as mp

from lieops.linalg.misc import identifyPairs, get_orientation
from lieops.linalg.matrix import create_pq2xieta
from lieops.linalg.common import eigenspaces
from lieops.linalg.checks import relative_eq

from njet.functions import get_package_name

def diagonal2block(D, code, orientation=[], tol=1e-10, **kwargs):
    r'''
    Computes a unitary map U which will diagonalize a matrix D of the form
    
       D = diag(a, b, ..., -a, ..., -b, ...)
       
    to a block matrix B = U.transpose().conjugate()@D@U of the form
    
        /  0   W  \
    B = |         |
        \ -W   0  /
    
    where W = diag(a*1j, b*1j, ...). The signs of a, b, ... can be controlled by an optional orientation (see below).
    
    Parameters
    ----------
    D:
        A list of diagonal entries
        
    code:
        A code defining the matrix object output.
        
    tol: float, optional
        A small parameter to identify the pairs on the diagonal of D.
        
    orientation: list, optional
        A list of expected eigenvalues.
    
        Explanation:
        Giving an orientation may be necessary because the Jordan Normal Form (or diagonal form of a matrix) is only determined up to
        permuation of its blocks (here a pair of eigenvalues). Without any orientation this may result in inconsistent output, for example
        if a diagonal matrix is given as input in symplectic_takagi, symplectic_takagi calls this routine, and the output
        of symplectic_takagic then diagonalizes to a matrix in which the original order and signs of the input diagonal matrix
        have been changed (permuted or multiplied by a minus sign). To prevent this, one can provide a list of 'expected' eigenvalues here,
        which will therefore fix the correct order/orientation.
    
    Returns
    -------
    U:
        The unitary matrix described above.
    '''
    dim2 = len(D)
    assert dim2%2 == 0
    dim = dim2//2

    if len(orientation) > 0:
        orientation = list(orientation)
        omat = get_orientation(orientation + orientation, D, **kwargs) # orientation is copied twice here, because "get_orientation" is made to work with two general lists of the same length.
        
    # Step 1: Determine the pairs on the diagonal which should be mapped.
    pairs = identifyPairs(D, condition=lambda x, y: abs(x + y) < tol, **kwargs)
    
    # Step 2: Construct U, assuming that it will transform a matrix with the above order
    if code == 'numpy':
        U = np.eye(dim2, dtype=complex)
    if code == 'mpmath':
        U = mp.eye(dim2, dtype=complex)
        
    U2by2 = create_pq2xieta(2, code=code, **kwargs)
    column_indices = []
    for k in range(len(pairs)):
        i, j = pairs[k]

        if len(orientation) > 0:
            # Using
            # U2by2perm := matrix([[0, 1], [1, 0]])@U2by2
            # instead of the default U2by2 corresponds to an exchange of the two eigenvalue pairs.
            # In effect, U2by2perm can be
            # computed by adding a -1 as sign. Furthermore, the eigenvalues might be related by +/- i.
            # For this purpose we have constructed the orientation matrix 'omat' above, in which these
            # values are stored. We retreive the factors now:
            oindices = np.where(omat[:,i])[0] # These are the (two) indices of those values in the reference 'orientation' vector, to which the value D[i] can be transformed by a multiplication of one of the members of [+1, -1, +i, -i].
            column_indices += list(oindices)
            oindex = oindices[0] # To get the sign, we take the first of the two indices, because these indices correspond to two identical values in 'orientation'.
            sign = omat[oindex, i].imag
            # now it holds sign*1j*D[i] == orientation[oindex] (up to the given tolerance).
        else:
            sign = 1
            column_indices += [k, k + dim]
            
        U[i, i] = U2by2[0, 0]
        U[i, j] = U2by2[0, 1]*sign
        U[j, i] = U2by2[1, 0]
        U[j, j] = U2by2[1, 1]*sign
        
    # Step 3: Ensure that U maps to the desired block ordering.
    # Pair k should be mapped to the indices (k, k + dim) and (k + dim, k). This can be done with a
    # transformation T as follows.
    if code == 'numpy':
        T = np.zeros([dim2, dim2])
    if code == 'mpmath':
        T = mp.zeros(dim2)

    k = 0
    for k in range(len(pairs)):
        i, j = pairs[k]
        k1 = column_indices[2*k]
        k2 = column_indices[2*k + 1]
        # map ei to ek1 and ej to ek2
        T[k1, i] = 1
        T[k2, j] = 1
        k += 1
        
    return U@T.transpose()

def anti_diagonalize_real_skew(M, **kwargs):
    r'''
    Anti-diagonalize a real skew symmetric nxn-matrix M so that
      R.transpose()@M@R 
    has skew-symmetric antidiagonal form with respect to (n//2 x n//2) block matrices:
    
             /  0   X  \
        M =  |         |
             \ -X   0  /
    
    where X is a diagonal matrix with positive entries and R is an orthogonal matrix.
    
    Attention: No check is made if the given matrix has real coefficients or is skew symmetric.
    
    Parameters
    ----------
    M:
        list of vectors defining a real skew symmetric (n x n)-matrix.

    **kwargs 
        Additional arguments are passed to .checks.relative_eq routine. Warning: If chosing 'mpmath' as code and dps value is
        set too small (corresponding to a too rough precision), then the tolerance 'tol' may have to be increased as well.
    
    Returns
    -------
    matrix
        Orthogonal real matrix R with the above property.
    '''
    
    code = get_package_name(M)
    evalues, evectors = eigenspaces(M, flatten=True, **kwargs)
    n = len(evalues)
    
    if code == 'numpy':
        sqrt2 = np.sqrt(2)
    if code == 'mpmath':
        if 'dps' in kwargs.keys(): 
            mp.mp.dps = kwargs['dps']
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
            same_plane = relative_eq(evalues[i].imag, -evalues[j].imag, **kwargs) # TODO: use newer routine identifyPairs instead
            if not same_plane:
                continue
                
            processed_indices.append(i)
            processed_indices.append(j)
            
            # Select the index belonging to the eigenvalue with positive imaginary part
            # Attention: This sets the signature of the matrix Omega in Eq. (6) in Williamson decomposition.
            # We want that the top right entry is positive and the bottom left entry is negative. Therefore:
            if evalues[i].imag >= 0: # the imaginary part of the eigenvalue of aj + 1j*bi (see below) will be positive
                pos_index = i
            else: # change the role of i and j (orientation) to maintain that ai + 1j*bi (see below) is positive; evalues[j].imag > 0
                pos_index = j
            
            ai = [(evectors[pos_index][k] + evectors[pos_index][k].conjugate())/sqrt2 for k in range(n)] # the 1/sqrt2-factor comes from the fact that evectors[pos_index] has been normalized to 1.
            bi = [-1j*(evectors[pos_index][k] - evectors[pos_index][k].conjugate())/sqrt2 for k in range(n)]
            v_block1.append(ai)
            v_block2.append(bi)
            
    out = np.array(v_block1 + v_block2).transpose()
    if code == 'mpmath':
        out = mp.matrix(out)
    return out

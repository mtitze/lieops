from tqdm import tqdm
import numpy as np
import warnings

from .lie import poly, lexp
from lieops.linalg.bch import bch_2x2
from lieops.ops.tools import poly2ad, ad2poly

def hadamard2d(*hamiltonians, keys, exact=False, **kwargs):
    '''
    Rearrange the terms in a sequence of Hamiltonians according to Hadamard's theorem:

    Consider a sequence of Hamiltonians *)
       h0, h1, h2, h3, h4, h5, ...
    By Hadamard's theorem, the sequence is equivalent to
       exp(h0)h1, h0, h2, h3, h4, h5, ...
    Continuing this argument, we could write
       exp(h0)h1, exp(h0)h2, h0, h3, h4, h5, ...
    and so on, so that we reach
       exp(h0)h1, exp(h0)h2, exp(h0)h3, ..., exp(h0)hn, h0
       
    *)
    I.e. the map exp(:h0:) o exp(:h1:) o exp(:h2:) o ..., but here -- for brevity --
    we do not write the exp-notation for the "base" maps.

    Instead of applying exp(h0) to every entry, we can perform this procedure with every 2nd
    term of the sequence:

       exp(h0)h1, h0, h2, h3, h4, h5, ...

       exp(h0)h1, h0, exp(h2)h3, h2, h4, h5, ...

       exp(h0)h1, exp(h0)exp(h2)h3, h0, h2, h4, h5, ...

       exp(h0)h1, exp(h0)exp(h2)h3, h0, h2, exp(h4)h5, h4, ...

       exp(h0)h1, exp(h0)exp(h2)h3, exp(h0)exp(h2)exp(h4)h5, h0, h2, h4, ...

    If not every second entry, but instead a list like h0, h2, h3 and h5 are of interest, then:

       exp(h0)h1, h0, h2, exp(h3)h4, h3, h5, ...

       exp(h0)h1, exp(h0)exp(h2)exp(h3)h4, h0, h2, h3, h5, ...

    In this routine the Hamiltonians are distinguished by two families defined by keys.
    Family one will be treated as the operators h0, h2, h3, h5, ... in the example above,
    which will be exchanged with members of family two.
    
    !!! Attention !!! 
    Only polynomials of dim 1 (i.e. 2D phase spaces) are supported at the moment.
    
    Parameters
    ----------
    hamiltonians: poly object(s)
        The Hamiltonians to be considered.
        
    keys: list
        A list of keys to distinguish the first group of Hamiltonians against the second group.
        
    exact: boolean, optional
        Whether to distinguish the Hamiltonians of group 1 by the given keys (True) or a subset of the given keys (False).
        
    **kwargs
        Optional keyworded arguments passed to the lexp.calcFlow routine.
    
    Returns
    -------
    list
        A list of the Hamiltonians [exp(h0)h1, exp(h0)exp(h2)exp(h3)h4, ...] as in the example above.
        
    list
        A list of polynomials representing the chain of the trailing operator h0#h2#h3#h5, ...
        
    list
        A list of polynomials representing the operators
        h0, h0#h2, h0#h2#h3, h0#h2#h3#h5, ...
    '''
    current_g1_operator = []
    g1_operators = []
    new_hamiltonians = []
    for hamiltonian in tqdm(hamiltonians, disable=kwargs.get('disable_tqdm', False)):
        
        if exact:
            condition = hamiltonian.keys() == set(keys)
        else:
            condition = set(hamiltonian.keys()).issubset(set(keys))
        
        if condition and hamiltonian != 0:
            # in this case the entry k belongs to group 1, which will be exchanged with the
            # entries in group 2.
            hamiltonian_ad = poly2ad(hamiltonian)
            if len(current_g1_operator) == 0:
                current_g1_operator = hamiltonian_ad
            else:
                current_g1_operator = bch_2x2(current_g1_operator, hamiltonian_ad)
            g1_operators.append(current_g1_operator)
        else:
            if len(current_g1_operator) == 0:
                new_hamiltonians.append(hamiltonian)
            else:
                op = lexp(ad2poly(current_g1_operator), **kwargs)
                new_hamiltonians.append(op(hamiltonian, **kwargs))
    if len(current_g1_operator) == 0 or len(new_hamiltonians) == 0:
        warnings.warn(f'No operators found to commute with, using keys: {keys}.')
    return new_hamiltonians, [ad2poly(current_g1_operator)], [ad2poly(op) for op in g1_operators]



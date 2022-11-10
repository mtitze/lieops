import numpy as np

import lieops.ops.poly

class yoshida:
    
    def __init__(self, scheme=[1, 1/2]):
        '''
        Model a symplectic integrator which is symmetric according to Yoshida [1].
        
        Parameters
        ----------
        scheme: list, optional
            A list of length 2 defining the 2nd order symmetric symplectic integrator.
            If scheme = [s1, s2], then the integrator is assumed to have the form
            exp(h(A + B)) = exp(h*s2*A) o exp(h*s1*B) o exp(h*s2*A) + O(h**2)
            By default, the "leapfrog" scheme [1, 1/2] is used.
        
        References
        ----------
        [1] H. Yoshida: "Construction of higher order symplectic integrators", 
        Phys. Lett. A 12, volume 150, number 5,6,7 (1990).
        '''
        self.scheme = scheme
        
    @staticmethod
    def branch_factors(m: int, scheme=[1, 1/2]):
        '''
        Compute the quantities in Eq. (4.14) in Ref. [1].
        '''
        if m == 0:
            return scheme
        else:
            frac = 1/(2*m + 1)
            z0 = -2**frac/(2 - 2**frac)
            z1 = 1/(2 - 2**frac)
            return z0, z1
        
    def build(self, n: int):
        '''
        Construct the coefficients for the symmetric Yoshida integrator according
        to Eqs. (4.12) and (4.14) in Ref. [1].
        
        Parameters
        ----------
        n: int
            The order of the integrator.
        '''
        z0, z1 = self.branch_factors(m=n, scheme=self.scheme)
        steps_k = [z1, z0, z1]
        for k in range(n):
            new_steps = []
            for step in steps_k:
                z0, z1 = self.branch_factors(m=n - k - 1, scheme=self.scheme)
                new_steps += [z1*step, z0*step, z1*step]
            steps_k = new_steps
            
        # In its final step, steps_k has the form
        # [a1, b1, a1, a2, b2, a2, a3, b3, a3, ..., bm, am]
        # where the aj's belong to the first operator and the bj's belong to the second operator.
        # Therefore, we have to add the inner aj's together. They belong to the index pairs
        # (2, 3), (5, 6), (8, 9), (11, 12), ...
        pair_start_indices = [j for j in range(2, len(steps_k) - 3, 3)]
        out = []
        k = 0
        while k < len(steps_k):
            if k in pair_start_indices:
                out.append(steps_k[k] + steps_k[k + 1])
                k += 2
            else:
                out.append(steps_k[k])
                k += 1
        return out
    
################
# General tools
################
    
def get_scheme_ordering(scheme):
    '''
    For a Yoshida-decomposition scheme obtain a list of indices defining
    the unique operators which have been created.
    '''
    # It is assumed that the given scheme defines an alternating decomposition of two operators. Therefore:
    scheme1 = [scheme[k] for k in range(0, len(scheme), 2)]
    scheme2 = [scheme[k] for k in range(1, len(scheme), 2)]
    unique_factors1 = np.unique(scheme1).tolist() # get unique elements, but maintain order (see https://stackoverflow.com/questions/12926898/numpy-unique-without-sort)
    unique_factors2 = np.unique(scheme2).tolist()
    indices1 = [unique_factors1.index(f) for f in scheme1]
    indices2 = [unique_factors2.index(f) for f in scheme2]
    indices = []
    for k in range(len(scheme)):
        if k%2 == 0:
            indices.append(indices1[k//2])
        else:
            indices.append(indices2[(k - 1)//2] + max(indices1) + 1) # we add max(indices1) + 1 here to ensure the indices for operator 2 are different than for operator 1
    
    # Relabel the indices so that the first element has index 0 etc.
    max_index = 0
    index_map = {}
    for ind in indices:
        if ind in index_map.keys():
            continue
        else:
            index_map[ind] = max_index
            max_index += 1
    return [index_map[ind] for ind in indices]

    
def combine_equal_hamiltonians(hamiltonians):
    '''
    Combine Hamiltonians which are adjacent to each other and admit the same keys.
    '''
    n_parts = len(hamiltonians)
    new_hamiltonians = []
    k = 0
    while k < n_parts:
        part_k = hamiltonians[k]
        new_part = part_k
        for j in range(k + 1, n_parts):
            part_j = hamiltonians[j]
            if part_k.keys() == part_j.keys():
                new_part += part_j
                k = j
            else:
                break
        k += 1
        new_hamiltonians.append(new_part)
    return new_hamiltonians

def split_by_order(hamiltonian, scheme):
    '''
    Split a Hamiltonian according to its orders.
    '''
    maxdeg = hamiltonian.maxdeg()
    mindeg = hamiltonian.mindeg()
    hom_parts = [hamiltonian.homogeneous_part(k) for k in range(mindeg, maxdeg + 1)]
    hamiltonians = [hamiltonian]
    for k in range(len(hom_parts)):
        keys1 = [u for u in hom_parts[k].keys()]
        new_hamiltonians = []
        for e in hamiltonians:
            new_hamiltonians += [h for h in e.split(keys=keys1, scheme=scheme) if h != 0]
        hamiltonians = new_hamiltonians
    return combine_equal_hamiltonians(new_hamiltonians) # combine_equal_hamiltonians is necessary here, because otherwise there may be adjacent Hamiltonians having the same keys, using the above algorithm.

######################################################
# Recursively split a hamiltonian into its monomials #
######################################################

def recursive_monomial_split(hamiltonian, scheme, include_values=True, **kwargs):
    '''
    Split a Hamiltonian into its monomials according to a given scheme.
    The scheme hereby defines a splitting of a Hamiltonian into alternating operators.
    This scheme will be applied recursively.
    
    Parameters
    ----------
    hamiltonian: poly
        A poly object (or dictionary) representing the Hamiltonian to be split.
        
    scheme: list
        A list of floats to define an alternating splitting.
        
    include_values: boolean, optional
        If True, then include the individual coefficients in front of the monomials in the final result.
        If False, then the initial individual coefficients are set to 1. This allows to conveniently 
        obtain the factors coming from the splitting routine.
        
    **kwargs
        Optional keyworded arguments passed to the internal routine _recursive_monomial_split.
        
    Returns
    -------
    list
        A list of dictionaries representing the requested splitting.
    '''
    # Preparation
    if include_values:
        splits = [{key: hamiltonian[key] for key in hamiltonian.keys()}]
    else:
        splits = [{key: 1 for key in hamiltonian.keys()}]
        
    split_result = _recursive_monomial_split(*splits, scheme=scheme, **kwargs)
    return [lieops.ops.poly(values=sr, dim=hamiltonian.dim, max_power=hamiltonian.max_power) for sr in split_result]


def _recursive_monomial_split(*splits, scheme, key_selection=lambda keys: [keys[j] for j in range(int(np.ceil(len(keys)/2)))]):
    '''
    Parameters
    ----------
    key_selection: callable, optional
        Function to map a given list of keys to a sublist, determining how to split
        the keys of a given Hamiltonian.
        
        For example:
        1. key_selection = lambda keys: [keys[0]]
           This will separate the first mononial from the others.
        2. key_selection = lambda keys: [keys[j] for j in range(int(np.ceil(len(keys)/2)))]
           This will separate the first N/2 (ceil) keys from the others. This may produce more
           terms than case 1. for small schemes, but may have better performance for larger schemes.
    '''

    new_splits = []
    iteration_required = False
    for split in splits:
        keys = list(split.keys())
        if len(keys) > 1:
            keys1 = key_selection(keys)
            keys2 = [k for k in keys if k not in keys1]
            # assert len(keys1) > 0 and len(keys2) > 0
            for k in range(len(scheme)):
                if k%2 == 0:
                    new_splits.append({k1: split[k1]*scheme[k] for k1 in keys1})
                else:
                    new_splits.append({k2: split[k2]*scheme[k] for k2 in keys2})
            iteration_required = True
        else:
            new_splits.append(split)
            
    if iteration_required:
        return _recursive_monomial_split(*new_splits, scheme=scheme)
    else:
        return new_splits
    
##################################################
# Algorithms to find sets of commuting monomials #
##################################################
# Codes may be dedicated to special folder in the future

### Algorithm 1

def get_commuting_parts1(monomials):
    '''
    Obtain a list of lists, each containing indices of the given monomials which
    commute with each other.
    '''
    mtab = _get_commuting_table(monomials)
    M = len(monomials)
    return _get_parts(M, mtab) 
    
def _get_commuting_table(monomials):
    '''
    Return a list of sets containing two indices each. If a set {i, j} appear in that list, this means
    that elements j and k do not commute.
    '''
    # Determine the 'multiplication table' of the given monomials
    powers = [list(m.keys())[0] for m in monomials]
    dim = len(powers[0])//2    
    partition1, partition2 = [], []
    for k in range(1, len(monomials)):
        for l in range(k):
            powers1, powers2 = powers[k], powers[l]
            if any([powers1[r]*powers2[r + dim] - powers1[r + dim]*powers2[r] != 0 for r in range(dim)]):
                partition1.append(k)
                partition2.append(l)
    return [set([i, j]) for i, j in zip(partition1, partition2)] # if set([i, j]) in this list, then element i and j will not commute.
    
def _get_parts(M, mtab):
    # Determine a list of objects. Each object corresponds to a list of indices of those elements which commute with each other.
    chains = []
    for k in range(M):
        # if k already appears in a previous chain, move to the next k
        cont = False
        for b in chains:
            if k in b:
                cont = True
                break
        if cont:
            continue
            
        # k did not yet appear in any previous chain. We create a new chain
        current_chain = [k]
        chains_k = [] # chains_k will contain those chains containing the index k and having length >= 2
        for l in range(M):
            if l == k:
                continue
                
            # if l does not commute with k, continue with the next l
            if set([k, l]) in mtab:
                if len(current_chain) > 1: # before we continue, add the current chain to the chains
                    chains_k.append(current_chain)
                    current_chain = [k]
                continue
                
            # add l to any previous chain if it commutes with those elements
            for ck in chains_k:
                if l not in ck and not any([set([j, l]) in mtab for j in ck]):
                    ck.append(l)                  

            # check if element l also commutes with all the other elements in the current chain. If so, then append l to current chain.
            if not any([set([j, l]) in mtab for j in current_chain]):
                current_chain.append(l)
            
            # if we reach the end of the loop, also add the current chain to the chains
            if l == M - 1 and len(current_chain) > 1:
                chains_k.append(current_chain)
                
        chains += chains_k
        
    return chains

### Algorithm 2

def get_commuting_parts2(monomials):
    mtab = _get_commuting_table(monomials)
    M = len(monomials)
    comm = {N: [c for c in range(M) if set({N, c}) not in mtab] for N in range(M)}
    parts = []
    for k in range(M):
        parts.append(_propagate_leafs(comm, M, k))
    return parts

def _get_indices_oi(comm, j, exclude, include):
    max_elements = 2
    k_of_interest = []
    intersections = []
    for k in include:
        if k == j or k in exclude:
            continue
        intersection = set(comm[j]).intersection(set(comm[k])).intersection(include)
        n_elements = len(intersection)
        if n_elements > max_elements:
            max_elements = n_elements
            k_of_interest = [k]
            intersections = [intersection]
        elif n_elements == max_elements:
            k_of_interest.append(k)
            intersections.append(intersection)
    return k_of_interest, intersections

def _propagate_leafs(comm, M, start):

    exclude = [start] # a list containing those elements which we already know that they are mutually commuting
    include = set(range(M)) # a list containing those elements which we want to chose the next mutually commuting element

    jj = start
    while len(include) != len(exclude): # TODO: Needs to be fully understood. If not, change to condition "len(j1) > 0"
        j1, ww = _get_indices_oi(comm, jj, exclude=exclude, include=include)

        # at each iteration there can be several indices of interest (or 'branches' 
        # of the same size). By choosing an index here we look at one of these branches of interest. 
        # We do not know if this choice leads to a chain (leaf) of maximal length in any circumstance.
        branch_index = 0
        jj = j1[branch_index]
        include = include.intersection(ww[branch_index])

        exclude.append(jj)
        
    return exclude

import numpy as np
import heapq
import warnings

import lieops.ops.poly

class yoshida:
    
    def __init__(self, start=[1, 1/2]):
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
        self.start = start
        
    @staticmethod
    def branch_factors(m: int, start=[1, 1/2]):
        '''
        Compute the quantities in Eq. (4.14) in Ref. [1].
        '''
        if m == 0:
            return start
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
        z0, z1 = self.branch_factors(m=n, start=self.start)
        steps_k = [z1, z0, z1]
        for k in range(n):
            new_steps = []
            for step in steps_k:
                z0, z1 = self.branch_factors(m=n - k - 1, start=self.start)
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
    
#################
# General tools #
#################
    
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

def combine_adjacent_hamiltonians(hamiltonians):
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

# Find sets of commuting monomials
# ================================

def get_commuting_parts(monomials, minimal=True, **kwargs):
    '''
    Obtain a list of lists, each containing indices of the given monomials which
    commute with each other.
        
    Parameters
    ----------
    minimal: boolean, optional
        If True, then return a set of indices which cover every element.
        If False, return a list of indices for every element (i.e. the return list has length = len(monomials)).
        In this case the code may yield several entries representing the same combinations.
        
    **kwargs
        Optional keyworded arguments passed to _propagate_branches routine.
    
    Returns
    -------
    A list of the same length as the given monomials. Each entry at position k 
    is a proposed list of indices for the monomials which commute with monomial k.
    '''
    p1, p2 = _get_commuting_table(monomials)
    p1, p2 = np.array(p1), np.array(p2)
    M = len(monomials)
    get_comm = lambda k: set(range(M)).difference(set(p2[p1 == k]).union(set(p1[p2 == k])))
    comm = [get_comm(k) for k in range(M)]
    # comm[k] is the set of elements which commute with element k. They do not necessarily commute with each other, but commutation with k is guaranteed.
    parts = []
    covering = []
    for k in range(M):
        
        # if k is already contained in some 'maximal' leaf
        if minimal and k in covering:
            continue
            
        p = _propagate_branches(comm, k, **kwargs)
        parts.append(p)
        covering += p
    return parts

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
    return partition1, partition2

def _get_indices_oi(comm, j, exclude, include):
    '''
    For a given dictionary 'comm', mapping every index k to a list of indices which belong to elements which commute with k,
    find those indices which have the largest intersection of their commuting elements.
    
    This is used in our greedy algorithm, see
    https://en.wikipedia.org/wiki/Greedy_algorithm
    '''
    max_elements = 2
    k_of_interest = []
    intersections = []
    j_domain = comm[j].intersection(include)  # those entries which commute with j and should be considered.
    for k in include.difference(exclude): # n.b. j is in exclude by (+), so k != j.
        intersection = j_domain.intersection(comm[k])
        n_elements = len(intersection)
        if n_elements > max_elements:
            max_elements = n_elements
            k_of_interest = [k]
            intersections = [intersection]
        elif n_elements == max_elements:
            k_of_interest.append(k)
            intersections.append(intersection)
    return k_of_interest, intersections

def _propagate_branches(comm, start, branch_index=0):
    
    exclude = [start] # a list containing those elements which we already know that they are mutually commuting
    include = comm[start] # a set containing those elements which we want to chose the next mutually commuting element

    jj = start
    while len(include) > len(exclude): # To this condition: Let C be a set of mutually commuting elements with jj in C (our goal is to find such a C, which we also call 'leaf'). Then the code will determine the intersection comm[c] for c in C, which is just equal C. The possible values to compute this intersection are contained in the set 'include', while those which are already determined are given in 'exclude'. If both sets are equal, then the code terminates.  
        j1, ww = _get_indices_oi(comm, jj, exclude=exclude, include=include)
        
        # At each iteration there can be several indices of interest ('branches' containing a maximal size of 
        # commuting elements). By choosing an index here we follow into one of these branches. 
        # Other choices of maximal branches will lead to different solutions. It is not guaranteed, however, if such
        # a maximal branch will lead to a leaf of maximal length (in fact one can construct a counterexample).
        # This method thus follows a greedy algorithm strategy, see 
        # https://en.wikipedia.org/wiki/Greedy_algorithm
        jj = j1[branch_index]
        include = include.intersection(ww[branch_index])

        exclude.append(jj) # (+)
        
    return exclude

###################
# Splitting tools #
###################
    
def _iterative_split_gen(k: int, n: int, r={(1,)}, warn=True):
    '''
    Generator for iterative split. 
    
    At every step (iteration), split away the 0, 2, 4, ... - entries of a given expression. 
    
    Parameters
    ----------
    k: int
        Number of iterations.
        
    n: int
        Number of splits per iteration.
    '''
    # This routine may start to perform less effective for large k and n values, but this is unavoidable:
    # The number M of terms after applying a split for k unique items -- and using a scheme of length n -- is in the order of
    # M ~ (n//2)**(k - 1)*n 
    # Hereby n//2 denotes the number of branches at every step which are repeatedly split. The number of steps is k - 1 and
    # the last factor n denotes the number of final branches.
    # For a large Hamiltonian with e.g. around 40 non-commuting terms, using n = 3 gives a small number (M ~ 3), but for n = 7, which
    # is the 4th order Yoshida integrator, we get M ~ 3e19 terms, which can not be avoided. That's why this algorithm
    # can be used only for the basic [1/2, 1, 1/2]-scheme for a large non-trivial Hamiltonian.
    
    assert k >= 1
    if warn:
        # 1. Compute the expected number of splits for a user warning:
        a = np.floor(n/2) # the number of terms which are not split away at one step
        b = np.ceil(n/2) # the number of terms which are split away at one step
        n_terms = a**(k - 1)*n # the number of terms which were split at the last step
        # add the numbers which were split away in the course of the split up (but not including) the last step:   
        if a > 1:
            # Geometric sum times the number of terms which are split away:
            n_terms += (1 - a**(k - 1))/(1 - a)*b
        else:
            # at every step, b terms are split away
            n_terms += b*(k - 1)
        if n_terms > 1000:
            warnings.warn(f'The number of terms, if using {k} items and a scheme of length {n}, will be {int(n_terms)} ...')
    
    # 2. Compute the splits
    for x in range(k):
        r = {t + (l,) if t[-1]%2 == 1 and len(t) == x + 1 else t for l in range(n) for t in r}
        yield r
        
def split_iteratively(n, scheme):
    '''
    Split a list of integers 0, 1, 2, ..., n - 1 iteratively, using a given scheme.
    
    At each step, item k will be split away from the remaining k + 1, k + 2, ..., n - 1 items.
    
    Parameters
    ----------
    n: int
        The number of unique items to be split.
        
    scheme: list
        A list of coefficients to be used in the split.
    '''
    assert n > 0
    if n == 1:
        warnings.warn('Splitting of a non-splittable element requested.')
        return [(0, 1)]
    n = n - 1 # the number of split iterations required to split n elements.    
    split = []
    final_tree = list(_iterative_split_gen(k=n, n=len(scheme)))[-1]
    order = sorted(final_tree)
    for history in order:
        coeff = 1
        for l in history[1:]:
            coeff *= scheme[l]
        if len(history) < n + 1:
            index = len(history) - 2
        elif history[-1]%2 == 0:
            # the last indices which are split are "n - 1" and "n".
            index = n - 1
        else:
            index = n
        split.append((index, coeff))
    return split

# Split according to total orders
# ===============================

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
    return combine_adjacent_hamiltonians(new_hamiltonians) # combine_adjacent_hamiltonians is necessary here, because otherwise there may be adjacent Hamiltonians having the same keys, using the above algorithm.

# Split a hamiltonian into its monomials
# ======================================

def iterative_monomial_split(hamiltonian, scheme, include_values=True, **kwargs):
    '''
    Split a given Hamiltonian iteratively into its monomials, by applying a given scheme.
    
    Parameters
    ----------
    hamiltonian: poly
        A poly object representing the Hamiltonian to be split.
        
    scheme: list
        A list of coefficients describing the split.
        
    include_values: boolean, optional
        A boolean switch to include the original values of the monomials in the final split (default),
        or to simply return the coefficients.
    '''
    monomials = hamiltonian.monomials()
    n_monomials = len(monomials)
    if n_monomials == 1:
        split = [(0, 1)]
    else:
        split = split_iteratively(len(monomials), scheme=scheme)
    result = []
    for s in split:
        index, value = s
        if include_values:
            result.append(monomials[index]*value)
        else:
            power = list(monomials[index].keys())[0]
            result.append(lieops.ops.poly(values={power: value}, max_power=hamiltonian.max_power, dim=hamiltonian.dim))
    return result

def recursive_monomial_split(hamiltonian, scheme, include_values=True, **kwargs):
    '''
    Split a Hamiltonian recursively into its monomials according to a given scheme.
    The scheme hereby defines a splitting of a Hamiltonian into alternating operators.
    
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
        # If include_values == False, then we set every value to one, thereby collecting
        # the split-coefficients on the go while computing the overall split.
        splits = [{key: 1 for key in hamiltonian.keys()}]
        
    split_result = _recursive_monomial_split(*splits, scheme=scheme, **kwargs)
    return [lieops.ops.poly(values=sr, dim=hamiltonian.dim, max_power=hamiltonian.max_power) for sr in split_result]

def _recursive_monomial_split(*splits, scheme, key_selection=lambda keys: [keys[0]]):
    '''
    Parameters
    ----------
    key_selection: callable, optional
        Function to map a given list of keys to a sublist, determining how to split
        the keys of a given Hamiltonian.
        
        Examples:
        1. key_selection = lambda keys: [keys[0]]
           This will separate the first mononial from the others. This should provide
           similar results as the "iterative_monomial_split" routine.
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
    
# Split according to subsets of commuting monomials
# =================================================

def iterative_commuting_split(hamiltonian, scheme, combine=True, include_values=True):
    '''
    Split a given Hamiltonian iteratively into commuting parts, according to a given scheme.
    
    Parameters
    ----------
    hamiltonian: poly
        A poly object, modeling the Hamiltonian to be split.
        
    scheme: list
        A list of values according to which the splitting of the Hamiltonian should be done.
        If scheme = [a0, b0, a1, b1, ...], then the largest part A of the Hamiltonian H which mutually commute are
        split to exp(a0*A) o exp(b0*CA) o exp(a1*A) o exp(b1*CA) o ...
        where H = A + CA. The operators exp(bj*CA) are subsequently been split into their commuting parts in the same manner.
    
    combine: boolean, optional
        If set to true, combine the commuting parts to individual poly objects. If false, return
        Lie operators for every monomial.
        
    include_values: boolean, optional
        If True, then include the individual coefficients in front of the monomials in the final result.
        If False, then the initial individual coefficients are set to 1. This allows to conveniently 
        obtain the factors coming from the splitting routine.
        
    Returns
    -------
    list
        A list of poly objects, representing a compositional approximation of the given Hamiltonian
        as described above.
    '''
    monomials = hamiltonian.monomials()
    
    # 1. Find subsets of monomials which commute with each other
    parts = get_commuting_parts(monomials)
    
    # 2. Find a decomposition of those monomials for the entire Hamiltonian, using a greedy algorithm to find
    #    a disjoint covering from the above commuting parts 
    covering = _greedy_set_cover([set(p) for p in parts])
    disjoint_covering = []
    partial_covering = set()
    for cs in covering:
        disjoint_covering.append(cs - partial_covering)
        partial_covering.update(cs)
        
    # 3. Now recursively split the given Hamiltonian into these parts, starting with the largest parts first, which should be the very first elements in disjoint_covering, since we did apply a greedy algorithm.
    n_cover = len(disjoint_covering)
    if n_cover == 1:
        split = [(0, 1)]
    else:
        split = split_iteratively(len(disjoint_covering), scheme=scheme)
    hamiltonians = []
    for s in split:
        index, value = s
        monomial_indices = disjoint_covering[index] # all these monomials commute with each other
        
        if include_values:
            s_hamiltonians = [monomials[index]*value for index in monomial_indices]
        else:
            s_hamiltonians = []
            for index in monomial_indices:
                power = list(monomials[index].keys())[0]
                s_hamiltonians.append(lieops.ops.poly(values={power: value}, max_power=hamiltonian.max_power, dim=hamiltonian.dim))
            
        if combine:
            s_hamiltonians = [sum([h for h in s_hamiltonians])]
            
        hamiltonians += s_hamiltonians
    return hamiltonians

def _greedy_set_cover(subsets):
    '''
    For a given cover of a parent set, attempt to find a covering using the least number of sets, by using a greedy algorithm.
    
    Parameters
    ----------
    subsets, list
        A list of sets, describing a covering of a 'total' set X.
        
    Returns
    -------
    list
        A list of sets, corresponding to a subset of the input sets, corresponding to a "local" minimal covering of X.
    
    Taken & modified with some additional comments from:
    https://stackoverflow.com/questions/21973126/set-cover-or-hitting-set-numpy-least-element-combinations-to-make-up-full-set
    '''
    parent_set = set().union(*subsets)
    _max = len(parent_set)
    # create the initial heap. Note 'subsets' can be unsorted,
    # so this is independent of whether remove_redunant_subsets is used.
    heap = []
    for s in subsets:
        # Python's heapq lets you pop the *smallest* value, so we
        # want to use max - len(s) as a score, not len(s).
        # len(heap) is just proving a unique number to each subset,
        # used to tiebreak equal scores.
        # See also https://docs.python.org/3/library/heapq.html
        heapq.heappush(heap, [_max - len(s), len(heap), s]) # len(heap) grows within this loop.
    results = []
    result_set = set()
    while result_set < parent_set:
        # determine the best set to be added to the covering.
        best = []
        unused = []
        while heap:
            score, count, s = heapq.heappop(heap) # score = _max - len(s), the smaller the score, the bigger the set 's'.
            if not best: # start with the current (first) element 's' of the heap to be considered the 'best'
                best = [_max - len(s - result_set), count, s]
                continue
            # 'best[0]' corresponds to the number of elements in the total set minus those which are supposed to 
            # enlarge the current set 'result_set', using the current 'best' set 'best[2]'. This means that the 
            # smaller this number is, the better this set will be suited as a candidate of the next choice of the cover.
            #
            # 'score' is of the form "_max - len(s - some_previous_result_set)", according to (1) and (2) below.
            if score >= best[0]:
                # Then score = _max - len(s - some_previous_result_set) >= _max - len(s - result_set) = best[0], i.e.
                # with best[2] =: sb we have
                # len(sb - result_set) >= len(s - some_previous_result_set), and for every other set s2 in the heap it holds
                # score2 >= score, i.e.
                # _max - len(s2 - some_previous_result_set2) >= _max - len(s - some_previous_result_set), i.e.
                # len(s - some_previous_result_set) >= len(s2 - some_previous_result_set2), overall therefore:
                # len(sb - result_set) >= len(s - some_previous_result_set) >= len(s2 - some_previous_result_set2).
                #
                # Consequently we know that the rest of the heap cannot beat
                # the best score "len(sb - result_set)". So push the subset "s" back on the heap, use the best set "sb" and
                # stop this iteration.
                heapq.heappush(heap, [score, count, s]) # (1)
                break
            # Now it holds score < best[0], so that (as commented above) with the current 'best' set best[2] =: sb we have
            # len(s - some_previous_result_set) < len(sb - result_set).
            # We now check if len(s - result_set) is actually smaller than len(sb - result_set):
            score = _max - len(s - result_set) # (2)
            if score >= best[0]: # equivalent to: len(sb - result_set) >= len(s - result_set)
                unused.append([score, count, s]) # do not use the set s, but add the set s, including its "updated" score, to the heap.
            else:
                # append the currently selected best set to the unused ones (including its "updated" score) & set the new 'best' value
                unused.append(best)
                best = [score, count, s]
        add_set = best[2]
        results.append(add_set)
        result_set.update(add_set)
        # subsets that were not the best get put back on the heap for next time.
        while unused:
            heapq.heappush(heap, unused.pop())
    return results

    
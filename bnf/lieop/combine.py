
from itertools import product
from scipy.special import bernoulli
from njet.jet import factorials

import numpy as np

'''
References:
[1]: S. P. Norsett, A. Iserles, H. Z. Munth-Kaas and A. Zanna: Lie Group Methods (2000).
[2]: T. Carlson: Magnus expansion as an approximation tool for ODEs (2005)
[3]: A. Iserles: Magnus expansions and beyond (2008)
'''


class tree:
    '''
    A tree according to Ref. [1]
    '''
    
    def __init__(self, *branches, time_power=0):
        self.branches = branches
        if len(branches) == 0:
            self.index = 1 # we set the index of the fundamental object to one, to have the easiest way to increase the index for higher-orders.
            self.pivot_branches = []
            self.factor = 1
            self.time_power = time_power # The power of the first coefficient in the Taylor expansion of the Hamiltonian with respect to time.
        else:
            assert len(branches) == 2
            self.index = branches[0].index + branches[1].index
            self.pivot_branches = [branches[0]] + branches[1].pivot_branches # The trees listen in Eq. (2.3), Ref. [3]
            
    def set_factor(self, b=[]):
        '''
        Set the factor associated with the coefficient of the tree.
        The factor will be given in self.factor.
        
        Attention: Calling this method requires that all the factors in self.pivot_branches have already been determined.
        
        Parameters
        ----------
        b: list, optional
            An optional argument so that the n-th element equals B_n/n!. If such a list is given,
            it must hold len(b) >= len(self.pivot_branches)
        '''
        n = len(self.pivot_branches)
        if len(b) == 0:
            f0 = bernoulli(n)[-1]/factorials(n)[-1]
        else:
            f0 = b[n]
            
        self.factor = f0
        for pbranch in self.pivot_branches:
            self.factor *= pbranch.factor
            
    def __eq__(self, other):
        if self.index != other.index:
            return False
        else:
            if self.index == 1: # == other.index
                return True
            else:
                return self.branches[0] == other.branches[0] and self.branches[1] == other.branches[1]
        
    def __str__(self):
        if len(self.branches) == 0:
            return 'H'
        else:
            return f'[I({str(self.branches[0])}), {self.branches[1]}]'
    
    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>'
    
    
def forests(k, time_power=0):
    '''
    Construct a set of trees with respect to a given index, according to Ref. [1].
    
    Parameters
    ----------
    k: int
        The maximal index up to which we want to construct the trees.
        
    time_power: int, optional
        Optional initial order of time, of the first Taylor coefficient of the given operator.
        
    Returns
    -------
    list
        A list where the entry j corresponds to a list containing all trees with index j.
    '''
    factors = bernoulli(k)/factorials(k)
    tree_groups = [[tree(time_power=time_power)]] # Representing the sets T_k in Ref. [2].
    forest_groups = [tree_groups[0]] # Representing the sets F_k in Ref. [2].
    for j in range(1, k + 1):
        # construct the set of Trees with respect to index j from trees of indices q < j and p < j so that q + p = j
        treesj = []
        for q in range((j - 1)//2 + 1): # the case j - q - 1 < q is already covered by q later on (elements of trees_q and trees_p are exchanged and added, if the provide a new unique tree).
            p = j - q - 1
            # N.B. q <= p
            for t1, t2 in product(tree_groups[q], tree_groups[p]):
                t12 = tree(t1, t2)
                t12.set_factor(factors)
                if t1 != t2:
                    t21 = tree(t2, t1)
                    t21.set_factor(factors)
                    time_power_12 = t1.time_power + t2.time_power + 1 # keep track of the power in time, see Ref. [2], Thm. 19
                    t21.time_power = time_power_12
                    t12.time_power = time_power_12
                    treesj.append(t21)
                else:
                    t12.time_power = t1.time_power + t2.time_power + 2 # keep track of the power in time, see Ref. [2], Thm. 19
                treesj.append(t12)
        tree_groups.append(treesj)
            
    time_powers = np.unique([t.time_power for tg in tree_groups for t in tg])
    max_power = k*(time_power + 1) + 2 # Trees of index k can only contribute to forests of k + 2 at most (if multiplied by a tree with index 1).
    # Each additional time_power can be extracted out of the brackets and therefore acts as a flat addition to this value. So we have max_power = k + 2 + k*time_power = k*(1 + time_power) + 2. We do not include forests beyond this value.
    forest_groups = {power: [t for tg in tree_groups for t in tg if t.time_power == power] for power in time_powers if power <= max_power}
    # N.B. in Ref. [2] it appears that the term belonging to "F_5" with coeff -1/24 is not correctly assigned. It should be in F_6. This code also predicts that it is in F_6. To be checked: In Ref. [1] #T_6 = 132 and #F_6 = 21, while here #T_6 = 136 and #F_6 = 21. Given the fact that there is agreement with #F_6, I am not sure whether there was a miscalculation of #T_6 in Ref. [1].
        
    return tree_groups, forest_groups


            
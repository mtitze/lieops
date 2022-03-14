
from itertools import product
from scipy.special import bernoulli
from njet.jet import factorials

'''
References:
[1]: T. Carlson: Magnus expansion as an approximation tool for ODEs (2005)
[2]: A. Iserles: Magnus expansions and beyond (2008)
'''


class tree:
    '''
    A tree according to Refs. [1, 2]
    '''
    
    def __init__(self, *branches):
        self.branches = branches
        if len(branches) == 0:
            self.index = 1 # we set the index of the fundamental object to one, to have the easiest way to increase the index for higher-orders.
            self.pivot_branches = []
            self.factor = 1
        else:
            assert len(branches) == 2
            self.index = branches[0].index + branches[1].index
            self.pivot_branches = [branches[0]] + branches[1].pivot_branches # The trees listen in Eq. (2.3), Ref. [2]
            
    def set_factor(self, b=[]):
        '''
        Set the factor associated with the coefficient of the tree.
        
        Attention: Assumes that all the factors in self.pivot_branches have already been determined.
        
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
            if self.index == 1: # then other.index == 1 as well
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
    
    
def tree_groups(k):
    '''
    Construct a set of trees with respect to a given index, according to Refs. [1, 2].
    
    Parameters
    ----------
    k: int
        The maximal index up to which we want to construct the trees.
        
    Returns
    -------
    list
        A list where the entry j corresponds to a list containing all trees with index j.
    '''
    factors = bernoulli(k)/factorials(k)
    out = [[tree()]]
    for j in range(1, k + 1):
        # construct the set of Trees with respect to index j from trees of indices q < j and p < j so that q + p = j
        treesj = []
        for q in range((j - 1)//2 + 1): # the case j - q - 1 < q is already covered by q later on (elements of trees_q and trees_p are exchanged and added, if the provide a new unique tree).
            p = j - q - 1
            # N.B. q <= p
            for t1, t2 in product(out[q], out[p]):
                t12 = tree(t1, t2)
                t12.set_factor(factors)                    
                treesj.append(t12)
                if t1 != t2:
                    t21 = tree(t2, t1)
                    t21.set_factor(factors)
                    treesj.append(t21)
        out.append(treesj)
    return out


            

from itertools import product

# Reference: [1] Carlson ...



class tree:
    '''
    A tree according to Ref. [1]
    '''
    
    def __init__(self, *branches):
        self.branches = branches
        if len(branches) == 0:
            self.index = 1 # we set the index of the fundamental object to one, to have the easiest way to increase the index for higher-orders.
        else:
            assert len(branches) == 2
            self.index = branches[0].index + branches[1].index
            
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
    
    
def build_trees(k):
    '''
    Construct a set of trees with respect to a given index, according to Ref. [1].
    
    Parameters
    ----------
    k: int
        The maximal index up to which we want to construct the trees.
        
    Returns
    -------
    list
        A list where the entry j corresponds to a list containing all trees with index j.
    '''
    out = [[tree()]]
    for j in range(1, k + 1):
        # construct the set of Trees with respect to index j from trees of indices q < j and p < j so that q + p = j
        treesj = []
        for q in range((j - 1)//2 + 1): # the case j - q - 1 < q is already covered by q later on (elements of trees_q and trees_p are exchanged and added, if the provide a new unique tree).
            p = j - q - 1
            # N.B. q <= p
            for t1, t2 in product(out[q], out[p]):
                treesj.append(tree(t1, t2))
                if t1 != t2:
                    treesj.append(tree(t2, t1))
        out.append(treesj)            
    
    return out